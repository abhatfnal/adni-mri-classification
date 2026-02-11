"""
Preprocess an MRI and related PET scans.
"""

import os
import sys
import argparse
import zipfile
import pandas as pd
import subprocess
import glob
import shutil 

def extract_dicom(zip_paths, scan_id, output_dir):
    extracted = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if scan_id in member and not member.endswith('/'):
                    filename = os.path.basename(member)
                    target_path = os.path.join(output_dir, filename)
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    extracted += 1
    return extracted

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('index', help='Index of the mri in the .csv file.')
    parser.add_argument('csv_path', help='Path to csv file containing mri and pet id pairs.')
    parser.add_argument('data_dir', help='Path to data dir where preprocessed files will be stored')
    parser.add_argument('zip_paths', help='Paths to zip files containing mri and pet scans.', nargs='+', default=[])

    args = parser.parse_args(sys.argv[1:])

    # Create data dir if not existent
    os.makedirs(args.data_dir, exist_ok=True)

    # Load csv file with pairs
    df = pd.read_csv(args.csv_path)

    # Convert index to integer
    idx = int(args.index)
    
    # Assuming there are less unique MRIs than PETs
    # Get unique MRI ids
    unique_mri_ids = sorted(df['mri_id'].unique())
    
    # Check that index is not out of bounds
    if idx >= len(unique_mri_ids):
        print("Error: index is out of bounds")
        sys.exit(0)

    # Otherwise get mri id
    mri_id = unique_mri_ids[idx]
    
    # Get PET ids corresponding to this MRI
    pet_ids = df[ df["mri_id"] == mri_id]["pet_id"].tolist()
    
    # Convert to strings and append I in front
    mri_id = "I" + str(mri_id)
    pet_ids = [ "I" + str(p) for p in pet_ids]
    
    print(f"Preprocessing MRI: {mri_id}, PETS: {pet_ids}")

    # ===== MRI Preprocessing =====
    try:
        
        # Get mri folder name
        mri_folder = os.path.join(args.data_dir, mri_id)

        # If folder doesn't exists already
        if not os.path.exists(mri_folder):
            
            # Create it
            os.makedirs(mri_folder)

            # Extract dicom files from zip files to folder
            n = extract_dicom(args.zip_paths, mri_id, mri_folder)
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
                raise RuntimeError(f"No NifTI found in {mri_folder}. Skipping pair.")
            
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
            
        else:
            print(f"MRI {mri_id} folder already found, skipping to PETs")
        
    except Exception as e:
        
        print(f"MRI preprocessing failed for {mri_id}: {repr(e)}. Exiting.")
        
        # Delete folder if it exists
        if os.path.exists(mri_folder):
            shutil.rmtree(mri_folder)
            
        # Exit gracefully
        sys.exit(0)
            
    
    # ===== PETs preprocessing
    
    # Required filepaths
    bias_corrected_path = os.path.join(mri_folder, 'm' + mri_id + '.nii')
    deformation_field_path = os.path.join(mri_folder, 'y_' + mri_id + '.nii')
    brain_mask_path = os.path.join(mri_folder, 'brain_mask_pet_m' + mri_id + '.nii')
        
    # Preprocess all PET corresponding to the specific MRI
    for pet_id in pet_ids:
        
        # Create pet folder
        pet_folder = os.path.join(args.data_dir, pet_id)
        
        # If it already exists, skip
        if os.path.exists(pet_folder):
            print(f"PET {pet_id} folder already found, skipping")
            continue
        
        # Otherwise create it
        os.makedirs(pet_folder)
        
        try:
            
            # Extract dicom files from zip files to folder
            n = extract_dicom(args.zip_paths, pet_id, pet_folder)
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
            subprocess.run(["matlab", "-batch", f"preprocess_pet('{pet_folder}','{bias_corrected_path}', '{deformation_field_path}', '{brain_mask_path}')"], check=True)
            
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
            
            # Skip to next PET
            continue
        
    # Delete extra files in MRI folder to save space
    try:
        to_delete = [ bias_corrected_path, deformation_field_path, brain_mask_path]
        for p in to_delete:
            os.remove(p)
            
    except:
        print(f"Failed to remove MRI temporary files.")
        
        

        
    
    
    