import pandas as pd
import os
import subprocess
import glob
import shutil
import random
import argparse
import sys
import zipfile 

def extract_dicom(zip_paths, scan_id, output_dir):
    """
    Extracts the .dcm files corresponding to the specified scan
    
    :param zip_paths: list of paths to zip files, where .dcm files are stored.
    :param scan_id: id of the scan.
    :param output_dir: output directory where .dcm files are extracted.
    """
    extracted = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if ('/' + scan_id + '/') in member and not member.endswith('/'):
                    filename = os.path.basename(member)
                    target_path = os.path.join(output_dir, filename)
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    extracted += 1
    return extracted


def preprocess_mri(mri_id, data_dir, zip_paths):

    # Get mri folder name
    mri_folder = os.path.join(data_dir, mri_id)

    try:

        # If folder already exists (and thus the MRI has already been processed), skip
        if os.path.exists(mri_folder):
            print(f"MRI {mri_id} folder already found, skipping")
            return mri_folder
            
        # Otherwise create it
        os.makedirs(mri_folder)

        # Extract dicom files from zip files to folder
        n = extract_dicom(zip_paths, mri_id, mri_folder)
        if n == 0:
            raise RuntimeError(f"No DICOMs extracted for {mri_id}")
            
        # Convert to .nii
        subprocess.run(
            ["dcm2niix", "-z", "n", "-o", mri_folder, mri_folder],
            check=True
        )

        # Remove .dcm files
        for p in glob.glob(os.path.join(mri_folder, "*.dcm")):
            os.remove(p)

        # Remove .json files
        for p in glob.glob(os.path.join(mri_folder, "*.json")):
            os.remove(p)

        # Get produced.nii files
        nii = glob.glob(os.path.join(mri_folder, "*.nii"))

        if len(nii) == 0:
            raise RuntimeError(f"No NifTI found in {mri_folder}. Skipping MRI {mri_id}")
        
        if len(nii) > 1:
            print(f"Expected exactly one NIfTI, found {len(nii)} in {mri_folder}")
            
            # Delete the others
            for p in nii[1:]:
                os.remove(p)

        # Rename .nii file. In case of multiple ones, just take the first one
        filename = os.path.join(mri_folder, mri_id + ".nii")
        os.replace(nii[0], filename)
        
        # Bias correction + segmentation + deformation field using Matlab script
        subprocess.run(["matlab", "-batch", f"preprocess_mri('{filename}')"], check=True)
    
        return mri_folder
    
    except Exception as e:
        print(f"MRI preprocessing failed for {mri_id}: {repr(e)}. Exiting.")
        
        # Delete folder if it exists
        if os.path.exists(mri_folder):
            shutil.rmtree(mri_folder)

        return None

def preprocess_pet(bias_corr_path, def_field_path, brain_mask_path, pet_id, data_dir, zip_paths):

    try:
        # Get pet folder
        pet_folder = os.path.join(data_dir, pet_id)
        
        # If it already exists, skip
        if os.path.exists(pet_folder):
            print(f"PET {pet_id} folder already found, skipping")
            return
        
        # Otherwise create it
        os.makedirs(pet_folder)
        
        # Extract dicom files from zip files to folder
        n = extract_dicom(zip_paths, pet_id, pet_folder)
        if n == 0:
            raise RuntimeError(f"No DICOMs extracted for {pet_id}")
        
        # Convert to .nii
        subprocess.run(
            ["dcm2niix", "-z", "n", "-o", pet_folder, pet_folder],
            check=True
        )

        # Remove .dcm files
        for p in glob.glob(os.path.join(pet_folder, "*.dcm")):
            os.remove(p)

        # Remove .json files
        for p in glob.glob(os.path.join(pet_folder, "*.json")):
            os.remove(p)

        # Run matlab preprocessing script
        subprocess.run(["matlab", "-batch", f"preprocess_pet('{pet_folder}','{bias_corr_path}', '{def_field_path}', '{brain_mask_path}')"], check=True)
        
        # Get name of produced .nii file
        nii = glob.glob(os.path.join(pet_folder, "*.nii"))
    
        # Rename it to clean_w_masked_rstatic_ID.nii
        filename = os.path.join(pet_folder, 'clean_w_masked_rstatic_' + pet_id + ".nii")
        os.replace(nii[0], filename)
        
    except Exception as e:
        print(f"PET preprocessing failed for {pet_id}: {repr(e)}. Skipping.")
        
        # Remove PET folder if it was created 
        if os.path.exists(pet_folder):
            shutil.rmtree(pet_folder)
    

def main(subject_index, csv_path, data_dir, zip_paths):

    # Set random seed 
    random.seed(42)

    # Read dataset csv
    df = pd.read_csv(csv_path)

    # Get subject id from subject index
    all_subjects = sorted(df["subject_id"].unique().tolist())
    subject_id = all_subjects[int(subject_index)]

    # Filter for scans belonging to the current patient
    df = df[df["subject_id"] == subject_id].copy()

    # Filter for MRIs and PETs
    df_mri = df[ df["group"].isin(["MRI-T1-3T", "MRI-T1-1.5T"]) ].copy()
    df_pet = df[ df["group"].apply( lambda s : s[:3] == "PET") ].copy()

    # If no MRI is available for this patient, return (PET needs MRI for preprocessing)
    if len(df_mri) == 0:
        return 

    # Keep track of mri_folders and ids for later use
    mri_info = []

    # Preprocess subject's MRIs first
    for index, row in df_mri.iterrows():

        mri_id = str(row["image_id"])
        mri_folder = preprocess_mri(mri_id=mri_id, data_dir=data_dir, zip_paths=zip_paths)

        if mri_folder is not None:
            mri_info.append((mri_id, mri_folder))
        

    # Preprocess subject's PETs
    for index, row in df_pet.iterrows():

        pet_id = str(row["image_id"])

        # Randomly pick one mri to use for the pet preprocessing.
        # They're all from the same patient, so it doesn't really matter which.
        mri_id, mri_folder = random.choice(mri_info)

        # Compose paths to bias corrected image, deformation field and brain mask
        bias_corr_path = os.path.join(mri_folder, 'm' + mri_id + '.nii')
        def_field_path = os.path.join(mri_folder, 'y_' + mri_id + '.nii')
        brain_mask_path = os.path.join(mri_folder, 'brain_mask_pet_m' + mri_id + '.nii')
        
        preprocess_pet(bias_corr_path, def_field_path, brain_mask_path, pet_id, data_dir, zip_paths)

    # Delete extra files in MRI folders to save space
    for mri_id, mri_folder in mri_info:

        # Delete extra files in MRI folder to save space
        try:
            bias_corr_path = os.path.join(mri_folder, 'm' + mri_id + '.nii')
            def_field_path = os.path.join(mri_folder, 'y_' + mri_id + '.nii')
            brain_mask_path = os.path.join(mri_folder, 'brain_mask_pet_m' + mri_id + '.nii')
            
            to_delete = [ bias_corr_path, def_field_path, brain_mask_path]
            for p in to_delete:
                os.remove(p)
                
        except:
            print(f"Failed to remove MRI temporary files.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('index', help='Index of the unique subject')
    parser.add_argument('csv_path', help='Path to csv file containing mri and pet ids.')
    parser.add_argument('data_dir', help='Path to data dir where preprocessed files will be stored')
    parser.add_argument('zip_paths', help='Paths to zip files containing mri and pet scans.', nargs='+', default=[])

    args = parser.parse_args(sys.argv[1:])

    print(args.index)
    print(args.csv_path)
    print(args.data_dir)
    print(args.zip_paths)

    main(args.index, args.csv_path, args.data_dir, args.zip_paths )