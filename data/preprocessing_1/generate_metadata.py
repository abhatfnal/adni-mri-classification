"""
Generates metadata .csv file for preprocessed .npy
"""

OUTPUT_DIR = './data'

import os
import pandas as pd

if __name__ == "__main__":
    
    # Load original metadata from abels
    df = pd.read_csv('../labels/adni_matched_nii_with_labels.csv')

    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    # Absolute path of output dir
    abs_path = os.path.abspath(OUTPUT_DIR)
    
    # New column (paths of .npy)
    new_col = []
    for i, row in df.iterrows():
        name, _ = os.path.splitext(os.path.basename(row['filepath']))
        new_col.append(abs_path+'/'+name+'.npy')
        
    # Add column for .npy path
    df['path'] = new_col

    df_out = df[['path','diagnosis']]

    df_out.to_csv(OUTPUT_DIR+'/'+'paths_labels.csv', index=False)

