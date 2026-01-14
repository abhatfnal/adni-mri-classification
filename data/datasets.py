import torch
import pandas as pd
import nibabel as nib
import numpy as np
import os

from torch.utils.data import Dataset

class ADNIDataset(Dataset):
    """
    ADNI dataset class
    """
        
    def __init__(self, csv_file, classes=[1,2,3], transform=None):
        
        self.transform = transform
        
        # Read csv
        self.data = pd.read_csv(csv_file)
        
        # Only keep rows with existing files
        self.data = self.data[ self.data['filepath'].apply( lambda x : os.path.exists(x))]
        
        # Only keep requested diagnoses
        self.data = self.data[self.data['diagnosis'].isin(classes)].copy()

        # Map them to 0, ... , |classes|
        diag_to_label = {diag: i for i, diag in enumerate(classes)}

        self.data['label'] = self.data['diagnosis'].map(diag_to_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vol = nib.load(row['filepath']).get_fdata().astype(np.float32)
   
        img = torch.from_numpy(vol).unsqueeze(0)  # → (1,D,H,W)

        if self.transform:
            img = self.transform(img)  # Tensor→Tensor on same device

        label = torch.tensor(int(row['label']), dtype=torch.long)
        return img, label
    
    def labels(self):
        return self.data['label'].astype(int).tolist()
    
    def groups(self):
        return self.data['ptid'].astype(str).tolist()
    
if __name__ == "__main__":
    
    dataset = ADNIDataset(
        csv_file="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_mri_pet/datasets/dataset_unimodal_mri_trainval.csv")
    
    index = 20
    print(dataset[index][0].shape)