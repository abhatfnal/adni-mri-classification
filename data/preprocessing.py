"""
    Parallalized preprocessing:
    
        - mri to diagnosis matching
        - metadata csv file generation
        - skull stripping
        - registration

"""

import os
import argparse
import pandas as pd
import numpy as np
import nibabel as nib

from datetime import datetime
from glob import glob
from scipy.ndimage import zoom
from fsl.wrappers import bet, flirt, LOAD
from settings import FSL_DIR, ADNI_ROOT_DIR, DIAGNOSIS_FILE, DATA_DIR

from concurrent.futures import ProcessPoolExecutor, as_completed


def match_mri_to_diagnosis():
    """
        Matches MRI scans with diagnosis of temporally closest exam
    """
    df_meta = pd.read_csv(DIAGNOSIS_FILE)
    df_meta = df_meta[['RID', 'EXAMDATE', 'DIAGNOSIS']]
    df_meta['EXAMDATE'] = pd.to_datetime(df_meta['EXAMDATE'], errors='coerce')
    df_meta.dropna(subset=['EXAMDATE'], inplace=True)
    
    # Prepare list for matches
    entries = []

    # Scan all .nii files
    root = ADNI_ROOT_DIR
    subjects = [os.path.join(root, s) for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))]

    for subj_path in subjects:
        subj_id = os.path.basename(subj_path)  # e.g. '002_S_1070'
        try:
            rid = int(subj_id.split('_')[-1])   # take last part (e.g. '1070')
        except ValueError:
            continue

        # Get only rows with this RID
        df_subj = df_meta[df_meta['RID'] == rid]

        for root_dir, dirs, files in os.walk(subj_path):
            for file in files:
                if file.endswith('.nii'):
                    full_path = os.path.join(root_dir, file)

                    # Extract scan date from path (e.g., '2006-12-18_09_11_48.0')
                    try:
                        date_str = full_path.split('/')[-3].split('_')[0]  # YYYY-MM-DD
                        scan_date = datetime.strptime(date_str, '%Y-%m-%d')
                    except Exception as e:
                        continue

                    # Match closest exam date (allow ±30 day tolerance)
                    df_subj.loc[:, 'days_diff'] = abs((df_subj['EXAMDATE'] - scan_date).dt.days)
                    match = df_subj[df_subj['days_diff'] <= 30]

                    if not match.empty:
                        exam = match.sort_values('days_diff').iloc[0]   # closest exam
                        
                        entries.append({
                            'filepath': full_path,
                            'rid': rid,
                            'scan_date': scan_date,
                            'exam_date': exam['EXAMDATE'],
                            'diagnosis': exam['DIAGNOSIS']
                        })

    # Save to CSV
    df_out = pd.DataFrame(entries)
    df_out.to_csv(os.path.join(DATA_DIR,'adni_matched_nii_with_labels.csv'), index=False)

    print(f" Done. Matched {len(df_out)} MRI scans with diagnosis labels.")


def generate_npy_metadata():
    """
        Generates metadata .csv file for preprocessed .npy
    """
    # Load original metadata
    df = pd.read_csv(os.path.join(DATA_DIR,'adni_matched_nii_with_labels.csv'))

    npy_dir =  os.path.join(DATA_DIR,'preprocessed_npy/')

    # Match .npy filenames based on same naming logic
    def build_npy_path(row, i):
        rid = row['rid']
        scan_date = pd.to_datetime(row['scan_date']).strftime('%Y%m%d')
        filename = f"sub-{rid}_date-{scan_date}_i-{i}.npy"
        return os.path.join(npy_dir, filename)

    # Add column for .npy path
    df['npy_path'] = [build_npy_path(row, i) for i, row in df.iterrows()]

    # Optional: keep only needed columns
    df_out = df[['npy_path', 'rid', 'scan_date','exam_date','diagnosis']]

    df_out.to_csv(os.path.join(DATA_DIR,'adni_preprocessed_npy_metadata.csv'), index=False)
    print(f"✅ Saved: adni_preprocessed_npy_metadata.csv with {len(df_out)} entries.")


def preprocess_and_save(i, rid, nii_path, scan_date, output_shape, output_dir):
    """
        Preprocesses single .nii file
    """
    
    # Make sure FSL output is correct
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI'
    
    # Template image path for registration
    TEMPLATE_IMG = os.path.join(FSL_DIR, 'data/standard/MNI152_T1_1mm_brain.nii.gz')
    
    try:
        # load
        img = nib.load(nii_path)

        # skull strip
        brain = bet(img, LOAD)['output']

        # register to MNI (uses global TEMPLATE_IMG)
        registration = flirt(
            src=brain,
            ref=TEMPLATE_IMG,
            out=LOAD,
            dof=12
        )['out']

        # z-score normalization
        data = registration.get_fdata().astype(np.float32)
        
        m, s = data.mean(), data.std()
        data = (data - m) / s if s > 0 else (data - m)

        # resize
        zoom_factors = [o / i for o, i in zip(output_shape, data.shape)]
        data = zoom(data, zoom_factors, order=1)

        # save
        date_str = pd.to_datetime(scan_date).strftime('%Y%m%d')
        out_name = f"sub-{rid}_date-{date_str}_i-{i}.npy"
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, data)
        return (out_path, None)
    except Exception as e:
        return (nii_path, str(e))


def run():
    """
        Default preprocessing using one thread
    """
    
    match_mri_to_diagnosis()
    generate_npy_metadata()
    
    output_shape = (128, 128, 128)
    output_dir = os.path.join(DATA_DIR,'preprocessed_npy/')
    
    df = pd.read_csv(os.path.join(DATA_DIR,'adni_matched_nii_with_labels.csv'))
    
    for i,row in df.iterrows():
        
        path, e = preprocess_and_save(i, 
                            row['rid'], 
                            row['filepath'], 
                            row['scan_date'], 
                            output_shape, 
                            output_dir)
        if e is None:
            print(f"Done {i}")
        else:
            print(f"Failed {i}: {e}")
        
    
    

def run_parallel():
    """
        Parallel preprocessing using multithreading
    """
    output_shape = (128, 128, 128)
    output_dir = os.path.join(DATA_DIR,'preprocessed_npy/')
    
    match_mri_to_diagnosis()
    generate_npy_metadata()
    
    df = pd.read_csv(os.path.join(DATA_DIR,'adni_matched_nii_with_labels.csv'))
    jobs = [(i, row['rid'], row['filepath'], row['scan_date'], output_shape, output_dir)
            for i,row in df.iterrows()]

    with ProcessPoolExecutor(max_workers=64) as exe:
        futures = {exe.submit(preprocess_and_save, *job): job for job in jobs}
        for f in as_completed(futures):
            path, err = f.result()
            if err:
                print(f"❌ Failed: {path} → {err}")
            else:
                print(f" {path} done!")
    print("✅ All done.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', action='store_true', help='run as multithreading cluster job')
    
    args = parser.parse_args()
    
    if args.job:
        run_parallel()
    else:
        run()