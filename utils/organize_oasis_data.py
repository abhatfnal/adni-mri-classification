import pandas as pd
import os
import shutil
import glob

# --- Configuration ---
CSV_FILE = '/project/aereditato/abhat/OASIS/OASIS_1/oasis_cross_sectional.csv'
SOURCE_DATA_DIR = '/project/aereditato/abhat/OASIS/OASIS_1/unpacked_data'
OUTPUT_DIR = '/project/aereditato/abhat/OASIS/OASIS_1/OASIS_1_Processed'
# --- End Configuration ---


def organize_data():
    """
    Reads clinical data, finds corresponding MRI scans, and copies them
    into new directories sorted by diagnostic class (AD or CN).
    """
    print("--- Starting Data Organization ---")

    ad_path = os.path.join(OUTPUT_DIR, 'AD')
    cn_path = os.path.join(OUTPUT_DIR, 'CN')
    os.makedirs(ad_path, exist_ok=True)
    os.makedirs(cn_path, exist_ok=True)
    print(f"Output directories created at: '{OUTPUT_DIR}'")

    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"❌ ERROR: Cannot find the CSV file '{CSV_FILE}'. Please check the name and path.")
        return

    ad_count = 0
    cn_count = 0
    missing_files = []

    for index, row in df.iterrows():
        subject_id = row['ID']
        cdr_score = row['CDR']

        # --- THIS IS THE FINAL LOGIC FIX ---
        # A subject is AD if CDR > 0. Otherwise (if CDR is 0.0 or NaN), they are CN.
        # Comparisons with NaN (e.g., NaN > 0) correctly evaluate to False.
        if cdr_score > 0:
            destination_folder = ad_path
            label = "AD"
        else:
            destination_folder = cn_path
            label = "CN"
        # --- END FIX ---
        
        search_pattern = os.path.join(SOURCE_DATA_DIR, '**', subject_id, 'PROCESSED', 'MPRAGE', 'SUBJ_111', '*_sbj_111.hdr')
        found_files = glob.glob(search_pattern, recursive=True)

        if not found_files:
            print(f"⚠️  WARNING: No scan found for subject {subject_id}")
            missing_files.append(subject_id)
            continue

        source_hdr_path = found_files[0]
        source_img_path = source_hdr_path.replace('.hdr', '.img')

        try:
            shutil.copy(source_hdr_path, destination_folder)
            shutil.copy(source_img_path, destination_folder)
            print(f"  -> Copied {label} subject: {subject_id}")
            
            if label == 'AD':
                ad_count += 1
            else:
                cn_count += 1
        except FileNotFoundError:
            print(f"⚠️  WARNING: Could not find .img file for {subject_id} at {source_img_path}")
            missing_files.append(subject_id)
            continue

    print("\n--- Data Organization Complete ---")
    print(f"Total subjects in CSV: {len(df)}")
    print(f"✅ Copied {cn_count} CN subject scans.")
    print(f"✅ Copied {ad_count} AD subject scans.")
    if missing_files:
        print(f"\n⚠️  Could not process {len(missing_files)} subjects (files may be missing):")
        print(sorted(list(set(missing_files))))


if __name__ == '__main__':
    organize_data()