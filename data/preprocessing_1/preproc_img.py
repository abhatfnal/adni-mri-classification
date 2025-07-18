"""
Applies fsl-bet, fsl-flirt, resize, zscore norm to 
mri given its index in the labels csv file. Saves to ./data.
"""

#====|| Config ||==========

TEMPLATE_IMG = '/project/aereditato/cestari/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'

output_shape = (128, 128, 128)

#==========================

if __name__ == "__main__":
    
    import os
    import sys
    import argparse
    import numpy as np
    import pandas as pd
    import nibabel as nib

    from fsl.wrappers import bet, flirt, LOAD
    from scipy.ndimage import zoom

    
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('path',help='Path to labels csv')
    parser.add_argument('index',help='Index of the MRI scan in the labels csv file')
    
    args = parser.parse_args(sys.argv[1:])
    
    try:
        # Read dataframe with paths 
        df = pd.read_csv(args.path)

        # Get mri index
        index = int(args.index)
        
        # Get mri path
        mri_path = df['filepath'][index]
        
        # Get file name for saving it later
        name, _ = os.path.splitext(os.path.basename(mri_path))
        
        # Load mri
        mri = nib.load(mri_path)
        
        # Apply fsl-bet (skull stripping)
        brain = bet(mri, LOAD)['output']

        # Apply fsl-flirt (register to MNI, uses TEMPLATE_IMG)
        registration = flirt(
            src=brain,
            ref=TEMPLATE_IMG,
            out=LOAD,
            dof=12
        )['out']

        # Z-score normalization
        data = registration.get_fdata().astype(np.float32)
        
        m, s = data.mean(), data.std()
        data = (data - m) / s if s > 0 else (data - m)

        # Resize
        zoom_factors = [o / i for o, i in zip(output_shape, data.shape)]
        data = zoom(data, zoom_factors, order=1)
        
        # Save to ./data
        np.save('./data/'+name, data)
        
    except Exception as e:
        print(f"Error: {e} ")