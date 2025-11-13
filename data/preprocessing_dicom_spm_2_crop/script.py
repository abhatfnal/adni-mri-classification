import os
import pandas as pd
import torchio as tio

csv_path = './dataset2.csv'
save_folder = './data_ADNI_2_3T_3D'

transform = tio.RandomAffine(
    scales=(1.7, 1.7),
    degrees=0,
    translation=0,
    p=1.0
)

#Load dataframe
df = pd.read_csv(csv_path)


for path  in df['filepath']:
    
    if path.endswith('.nii'):
        
        print(f'Processing {os.path.basename(path)}')
        
        sample = tio.ScalarImage(path)
        
        # Apply the transformation
        transformed_sample = transform(sample)
        
        # Save the transformed sample
        transformed_sample.save(os.path.join(save_folder, os.path.basename(path)))