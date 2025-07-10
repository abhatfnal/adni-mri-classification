import os
import pandas as pd
from datetime import datetime
from glob import glob

# Load metadata
meta_file = '/project/aereditato/abhat/ADNI/DXSUM_05Apr2025.csv'
df_meta = pd.read_csv(meta_file)
df_meta = df_meta[['RID', 'EXAMDATE', 'DIAGNOSIS']]
df_meta['EXAMDATE'] = pd.to_datetime(df_meta['EXAMDATE'], errors='coerce')
df_meta.dropna(subset=['EXAMDATE'], inplace=True)

# Prepare list for matches
entries = []

# Root directory containing merged unzipped 3T data
root = '/project/aereditato/abhat/ADNI/3T_Datasets/ADNI/merged_unzipped/ADNI'
subjects = [os.path.join(root, s) for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))]

for subj_path in subjects:
    subj_id = os.path.basename(subj_path)  # e.g. '002_S_1070'
    try:
        rid = int(subj_id.split('_')[-1])   # take last part (e.g. '1070')
    except ValueError:
        continue

    # Filter metadata for this RID
    df_subj = df_meta[df_meta['RID'] == rid].copy()

    for root_dir, dirs, files in os.walk(subj_path):
        for file in files:
            if file.endswith('.nii'):
                full_path = os.path.join(root_dir, file)

                # Extract scan date from folder structure (e.g., '2006-12-18_09_11_48.0')
                try:
                    scan_folder = full_path.split('/')[-3]  # folder name before Ixxxx
                    date_str = scan_folder.split('_')[0]
                    scan_date = datetime.strptime(date_str, '%Y-%m-%d')
                except Exception:
                    continue

                # Match closest exam date (±30 days tolerance)
                df_subj['days_diff'] = abs((df_subj['EXAMDATE'] - scan_date).dt.days)
                match = df_subj[df_subj['days_diff'] <= 30]

                if not match.empty:
                    diagnosis = match.sort_values('days_diff').iloc[0]['DIAGNOSIS']
                    entries.append({
                        'filepath': full_path,
                        'rid': rid,
                        'scan_date': scan_date,
                        'diagnosis': diagnosis
                    })

# Save to CSV
df_out = pd.DataFrame(entries)
df_out.to_csv('adni_matched_nii_with_labels.csv', index=False)

print(f"✅ Done. Matched {len(df_out)} MRI scans with diagnosis labels.")
