import os
import torchio as tio

nii_folder = '../preprocessing_dicom_spm_2/data/'
save_folder = './data'

transform = tio.RandomAffine(
    scales=(1.7, 1.7),
    degrees=0,
    translation=0,
    p=1.0
)

for nii_file in os.listdir(nii_folder):
    if nii_file.endswith('.nii'):
        
        print(f'Processing {nii_file}')
        
        nii_path = os.path.join(nii_folder, nii_file)
        sample = tio.ScalarImage(nii_path)
        
        # Apply the transformation
        transformed_sample = transform(sample)
        
        # Save the transformed sample
        transformed_sample.save(os.path.join(save_folder, nii_file))