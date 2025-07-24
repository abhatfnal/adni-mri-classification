"""
Plots intensity histogram of .nii image.

"""

import sys
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Plots intensity histogram of given .nii file')
    
    parser.add_argument('path',help='Path of .nii file')
    parser.add_argument('-b',help='Number of bins')
    
    args = parser.parse_args(sys.argv[1:])
    
    try:
        import numpy as np
        import nibabel as nib
        
        # Load .nii file, convert to numpy
        img = nib.load(args.path).get_fdata().astype(np.float32)
        
        # Check for NaNs and replace them with mean
        nan_mask = np.isnan(img)
        mean = np.nanmean(img)
        img[nan_mask] = mean
        
        print(f"Replaced { np.sum(nan_mask) } NaNs with mean")
        
        # Z-score normalization
        if img.std() != 0:
            img = (img - img.mean())/img.std()
        else:
            img = img - img.mean()
        
        # Produce histogram
        if args.b:
            bin_number = int(args.b)
        else:
            bin_number = 10
            
        hist,bins = np.histogram(img, bins=bin_number)
        
        print(hist)
        print(bins)
        
    except Exception as e:
        print(f"Error: {e}")