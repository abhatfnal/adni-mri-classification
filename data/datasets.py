import torch
import pandas as pd
import nibabel as nib
import numpy as np

from torch.utils.data import Dataset

class ADNIDataset(Dataset):
    """
    ADNI dataset class
    """
        
    def __init__(self, csv_file, transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Keep only relevant diagnoses and create labels
        # self.data = self.data[self.data['diagnosis'].isin([1.0, 2.0, 3.0])]
        # self.data['label'] = self.data['diagnosis'].astype(int) - 1
        
        # Keep only relevant diagnoses and create labels
        self.data = self.data[self.data['diagnosis'].isin([1.0, 2.0, 3.0])]
        unique_diagnoses = sorted(self.data['diagnosis'].unique())
        diag_to_label = {diag: i for i, diag in enumerate(unique_diagnoses)}
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
        return self.data['rid'].astype(int).tolist()