import torch
import pandas as pd
import nibabel as nib
import numpy as np
import os

from torch.utils.data import Dataset

from pkg.utils.diagnosis_matching import match_diagnosis
from pkg.utils.multimodality import create_multimodal_dataframe

from tqdm import tqdm


class ADNIDataset(Dataset):
    
    def __init__(self,
                 data_dir,
                 scan_csv, 
                 diagnostic_csv, 
                 modalities={
                     "MRI":["MRI-T1-3T", "MRI-T1-1.5T"],
                     "PET":["PET-FDG"] }, 
                 diagnosis=[1,2,3], 
                 tolerance=180,
                 verbose=2):
    
        self.data_dir =data_dir
        self.scan_csv = scan_csv
        self.diagnostic_csv = diagnostic_csv
        self.modalities = modalities 
        self.diagnosis = diagnosis
        self.tolerance = tolerance
        self.verbose = verbose

        
    def setup(self):

        # Load csv with all scans descriptions
        df_scan = pd.read_csv(self.scan_csv)

        # Filter for those available in the data dir
        available_scans = os.listdir(self.data_dir)
        df_scan = df_scan[ df_scan["image_id"].isin(available_scans)]

        # Create modalities from groups as described in the parameter "modalities"
        df_scan["modality"] = ""

        for mode, groups in self.modalities.items():
            df_scan.loc[df_scan["group"].isin(groups), "modality"] = mode 

        # Filter for specified modalities 
        df_scan = df_scan[ df_scan["modality"].isin(list(self.modalities.keys()))].copy()

        # Load csv with all visits
        df_diagnostic = pd.read_csv(self.diagnostic_csv)

        if self.verbose > 0:
            print("Matching scans to diagnoses...")

        # Match scans to visits, adding diagnosis column
        df_scan = match_diagnosis(df_scan, df_diagnostic, self.tolerance)

        # Filter for specified diagnosis
        df_scan = df_scan[ df_scan["diagnosis"].isin(self.diagnosis)].copy()
        
        if self.verbose > 0:
            print("Verifying scans available in dir...")

        # Add file paths
        paths = []
        for index, row in tqdm(df_scan.iterrows()):

            id = row["image_id"]
            allowed_filenames = ['clean_w_masked_m' + id + '.nii', 
                                 'clean_w_masked_rstatic_' + id + '.nii']
            
            found = False
            for fname in allowed_filenames:

                path = os.path.join( self.data_dir, id, fname )
                if os.path.exists(path):
                    paths.append(path)
                    found = True
                    break

            if not found:
                paths.append(None)

        df_scan["path"] = paths
        df_scan = df_scan[ df_scan["path"].notna()]

        if self.verbose > 0:
            print("Creating multimodal samples...")

        # Create multimodal samples
        df_multimodal = create_multimodal_dataframe(df_scan, tolerance=self.tolerance)

        # Add labels: map diagnosis to 0, ... , |classes|
        diag_to_label = {diag: i for i, diag in enumerate(self.diagnosis)}
        df_multimodal['label'] = df_multimodal['diagnosis'].map(diag_to_label)

        # Save
        self.df_scan = df_scan
        self.df_multimodal = df_multimodal
        
    def __len__(self):
        return len(self.df_multimodal)
    
    def __getitem__(self, index):

        scans = []
        row = self.df_multimodal.loc[index]

        for mode in sorted(list(self.modalities.keys())):
            
            if row[mode] is None:
                scans.append(None)
                continue

            path = self.df_scan.loc[ self.df_scan["image_id"] == row[mode], "path"].tolist()[0]
            vol = nib.load(path).get_fdata().astype(np.float32)         
            img = torch.from_numpy(vol).unsqueeze(0)  # Add channel dimension (1,D,H,W)

            #TODO: Add augmentation transforms !

            scans.append(img)
    
        X = tuple(scans) if len(self.modalities) > 1 else scans[0]
        y = torch.tensor(int(row['label']), dtype=torch.long)

        return X,y

    def groups(self):
        return self.df_multimodal["subject_id"].astype(str).tolist()

    def labels(self):
        return self.df_multimodal["label"].astype(int).tolist()

    def strat_keys(self):
        return self.df_multimodal["strat_key"].astype(str).tolist()

    
class TransformDataset(Dataset):
    """
    Simple class that applies a transform to a dataset
    """
    def __init__(self, base_ds, transform=None):
        
        self.base = base_ds
        self.transform = transform
        
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.transform is not None:
            x = self.transform(x)
            
            return x, y
        
        
# class ADNIDataset2(Dataset):
#     """
#     ADNI dataset class
#     """
        
#     def __init__(self, csv_file, classes=[1,2,3], transform=None):
        
#         self.transform = transform
        
#         # Read csv
#         self.data = pd.read_csv(csv_file)
        
#         # Only keep rows with existing files
#         self.data = self.data[ self.data['filepath'].apply( lambda x : os.path.exists(x))]
        
#         # Only keep requested diagnoses
#         self.data = self.data[self.data['diagnosis'].isin(classes)].copy()

#         # Map them to 0, ... , |classes|
#         diag_to_label = {diag: i for i, diag in enumerate(classes)}

#         self.data['label'] = self.data['diagnosis'].map(diag_to_label)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         vol = nib.load(row['filepath']).get_fdata().astype(np.float32)
   
#         img = torch.from_numpy(vol).unsqueeze(0)  # → (1,D,H,W)

#         if self.transform:
#             img = self.transform(img)  # Tensor→Tensor on same device

#         label = torch.tensor(int(row['label']), dtype=torch.long)
#         return img, label
    
#     def labels(self):
#         return self.data['label'].astype(int).tolist()
    
#     def groups(self):
#         return self.data['ptid'].astype(str).tolist()
    
if __name__ == "__main__":
    
    dataset = ADNIDataset(
        csv_file="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_mri_pet/datasets/dataset_unimodal_mri_trainval.csv")
    
    index = 20
    print(dataset[index][0].shape)