import os
import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
import nibabel as nib
import numpy as np

import pandas as pd
from pathlib import Path


class ADNIDataset(Dataset):
    """
    ADNI dataset class
    """
        
    def __init__(self, csv_file, transform=None, zscore_norm=True):
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.zscore_norm = zscore_norm

        # Keep only relevant diagnoses and create labels
        self.data = self.data[self.data['diagnosis'].isin([1.0, 2.0, 3.0])]
        self.data['label'] = self.data['diagnosis'].astype(int) - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        img = nib.load(row['filepath']).get_fdata().astype(np.float32)

        if self.transform:
            img = self.transform(img)
        
        # Replace Nans with global mean
        # TODO: Think how to handle better
        mean = np.nanmean(img)
        img[np.isnan(img)] = mean
        
        # Z-Score normalize
        if self.zscore_norm and img.std() != 0:
            img = (img - img.mean())/img.std()
        else:
            img = img.mean()
            
        # already normalized in preprocessing, just add channel
        img = np.expand_dims(img, 0)  # [1,D,H,W]

        # check for Nan
        if np.any(np.isnan(img)):
            print("Alarm: Nan in dataset!")
            
        return torch.from_numpy(img), torch.tensor(row['label'])
    
    def labels(self):
        return self.data['label'].astype(int).tolist()