"""
Replaces NaNs with mean value, performs intensity histogram matching and z-score normalization,
and saves output to a new file.
"""
import sys
import argparse
import nibabel as nib
import numpy as np
from skimage.exposure import match_histograms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Histogram-match and normalize NIfTI volumes')
    parser.add_argument('in_path', help='Input NIfTI file path (skull-stripped)')
    parser.add_argument('template_path', help='Reference NIfTI template path')
    parser.add_argument('out_path', help='Output NIfTI file path')
    args = parser.parse_args(sys.argv[1:])

    # Load image and template
    img_nii = nib.load(args.in_path)
    img = img_nii.get_fdata().astype(np.float32)
    template = nib.load(args.template_path).get_fdata().astype(np.float32)

    # Replace NaNs with mean
    mean_val = np.nanmean(img)
    img[np.isnan(img)] = mean_val

    # Intensity histogram matching
    img = match_histograms(img, template, channel_axis=None)

    # Z-score normalization
    if img.std() != 0:
        img = (img - img.mean()) / img.std()
    else:
        img = img.mean()

    # Save to new file
    out_nii = nib.Nifti1Image(img, img_nii.affine, img_nii.header)
    nib.save(out_nii, args.out_path)
    print(f"Saved intensity-normalized image to {args.out_path}")
