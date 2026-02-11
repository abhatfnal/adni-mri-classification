import pandas as pd
import os
import glob

# --- Configuration ---
# The directory where your final, clean AD and CN folders are located
FINAL_DATA_DIR = '/project/aereditato/abhat/OASIS/OASIS_1/OASIS_1_Final'
# The name of the output CSV file
OUTPUT_CSV = '/project/aereditato/abhat/OASIS/OASIS_1/oasis_1_master.csv'
# --- End Configuration ---

def create_master_csv():
    """
    Scans the final data directory and creates a master CSV with file paths,
    subject IDs, and diagnoses.
    """
    data = []
    # Find all .nii files in the AD and CN subdirectories
    for class_label, class_name in enumerate(['CN', 'AD']):
        class_path = os.path.join(FINAL_DATA_DIR, class_name)
        nii_files = glob.glob(os.path.join(class_path, '*.nii'))

        for file_path in nii_files:
            # Extract the base filename (e.g., 'OAS1_0001_MR1_mpr_n4_anon_sbj_111_preprocessed')
            base_name = os.path.basename(file_path).replace('_preprocessed.nii', '')
            
            # Extract the subject ID (e.g., 'OAS1_0001') from the filename
            # This assumes the subject ID is the first part of the filename, up to the second underscore
            try:
                subject_id = '_'.join(base_name.split('_')[:2])
            except IndexError:
                subject_id = base_name # Fallback for unexpected filenames

            data.append({
                'filepath': file_path,
                'subject_id': subject_id,
                'diagnosis': class_label
            })

    # Create a DataFrame and save it to CSV
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"--- Master CSV Creation Complete ---")
    print(f"Total scans found: {len(df)}")
    print(f"Class distribution:\n{df['diagnosis'].value_counts().sort_index()}")
    print(f"\n✅ Master CSV saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    create_master_csv()