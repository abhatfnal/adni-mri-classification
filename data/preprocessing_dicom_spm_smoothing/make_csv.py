"""
Compute ./data/datatset.csv file.
Contains paths to scans and its respective diagnosis.
"""
import os
import re
import sys
import pandas as pd
import argparse

from datetime import datetime

def parse_filename(filename):
    """
    Extracts patient ID, date from a DICOM filename.
    
    Parameters:
    filename (str): The DICOM filename string to parse.
    
    Returns:
    dict: A dictionary containing 'patient_id', 'date'
    """

    m = re.match(r'^ADNI_(?P<id>\d+_S_\d+)_.*?_(?P<date>\d{4}-\d{2}-\d{2})', filename)
    if m:
        patient_id = m.group('id')
        date = datetime.strptime(m.group('date'),"%Y-%m-%d") 

    rid = int(patient_id.split('_')[-1])

    return (rid, date)


def compute_labels(diagnosis_file, img_dir):
    """
    Get list of (patient_id,date) then matches with temporally closest diagnosis and image absolute path.
    Dumps data in img_dir/dataset.csv.
    """
    try:

        # file names
        nii_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.nii')]
        
        # get visits
        df_meta = pd.read_csv(diagnosis_file)
        df_meta = df_meta[['RID', 'EXAMDATE', 'DIAGNOSIS']]
        df_meta['EXAMDATE'] = pd.to_datetime(df_meta['EXAMDATE'], errors='coerce')
        df_meta.dropna(subset=['EXAMDATE'], inplace=True)
        
        # Prepare list for matches
        entries = []
    
        # Get subjects list (rids)
        subjects = list(set([parse_filename(f)[0] for f in nii_files]))
        
        print(f"Subjects {subjects}")
        print(f"files: {nii_files}")
        
        # Iterate over subjects
        for subj in subjects:
            
            # Get only rows with this RID
            df_subj = df_meta[df_meta['RID'] == subj]
            
            #Loop through images of current patient
            for f in nii_files:
                
                rid, scan_date = parse_filename(f)
                
                # if matching subject (same rid)
                if rid == subj:
                    
                    # Match closest exam date (allow ±30 day tolerance)
                    df_subj.loc[:, 'days_diff'] = abs((df_subj['EXAMDATE'] - scan_date).dt.days)
                    match = df_subj[df_subj['days_diff'] <= 30]
                    
                    # img absolute path
                    path = os.path.abspath( os.path.join(img_dir, f))
                    
                    if not match.empty:
                        exam = match.sort_values('days_diff').iloc[0]   # closest exam
                        
                        entries.append({
                            'filepath': path,
                            'rid': rid,
                            'scan_date': scan_date,
                            'exam_date': exam['EXAMDATE'],
                            'diagnosis': exam['DIAGNOSIS']
                        })
                    
            
        # Save to CSV
        df_out = pd.DataFrame(entries)
        df_out.to_csv(os.path.join(img_dir, 'dataset.csv'), index=False)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('diagnosis_file')
    parser.add_argument('img_dir')
    
    args = parser.parse_args(sys.argv[1:])
    
    compute_labels(args.diagnosis_file, args.img_dir)
    
    