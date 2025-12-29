import pandas as pd 
import os 
import zipfile

data_dir = './data'
df_path = './dataset.csv'

processed_ids = [ int(id[1:]) for id in list(set(os.listdir(data_dir)))]
df = pd.read_csv(df_path)

all_mri_ids = [int(id) for id in df["mri_id"].unique()]
all_pet_ids = [int(id) for id in df["pet_id"].unique()]

missing_mri = [ id for id in all_mri_ids if id not in processed_ids]
missing_pet = [ id for id in all_pet_ids if id not in processed_ids]

print(f"Missing MRI: {len(missing_mri)}")
print(f"Missing PET: {len(missing_pet)}")

# Print all missing PET scans
print("___________________")
for pet in missing_pet:
    print(pet)

print("___________________")

# There are no missing MRIs. 
# Check if missing PETs are present in zip files. 

# zip_paths = ['/project/aereditato/cestari/downloads/ADNI/Multimodal_MRI_PET_dataset_1.zip',
#              '/project/aereditato/cestari/downloads/ADNI/Multimodal_MRI_PET_dataset_2.zip']

# all_files = []

# for zip_path in zip_paths:
    
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        
#         all_files.append(zip_ref.namelist())
          
          
# found_count = 0
      
# for id in missing_pet:
    
#     for file in all_files:
#         if (str(id)) in file:
#             found_count += 1
#             print(f"Id: {id} found in file {file}")
#             break
             
            
# print(f"Found {found_count} files out of {len(missing_pet)} missing!")