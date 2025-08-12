import nibabel as nib
import numpy as np
import os

# directory containing your .nii files
data_dir = '/project/aereditato/cestari/adni-mri-classification/data/preprocessing_dicom_spm/data/'

sum_data = None
affine = None
header = None
file_list = []

# collect all filenames
for root, dirs, files in os.walk(data_dir):
    for fn in files:
        if fn.endswith('.nii') or fn.endswith('.nii.gz'):
            file_list.append(os.path.join(root, fn))

# accumulate
for idx, filepath in enumerate(file_list):
    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)
    if sum_data is None:
        sum_data = data
        affine   = img.affine         # grab the spatial transform
        header   = img.header.copy()  # grab a copy of the header
    else:
        sum_data += data

# compute average
avg_data = sum_data / len(file_list)

# create a new NIfTI image
avg_img = nib.Nifti1Image(avg_data, affine, header)

# save to disk
nib.save(avg_img, 'average_brain.nii')
print(f"Saved average image of shape {avg_data.shape} to 'average_brain.nii'")
