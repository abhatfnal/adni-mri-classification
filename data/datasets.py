import os
import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
import numpy as np

from .settings import DATA_DIR

CSV_FILE = os.path.join(DATA_DIR, 'adni_preprocessed_npy_metadata.csv')

class ADNIDataset(Dataset):
    """
    Full ADNI dataset class
    """
        
    def __init__(self, transform=None):
        
        self.data = pd.read_csv(CSV_FILE)
        self.transform = transform

        # Keep only relevant diagnoses and create labels
        self.data = self.data[self.data['diagnosis'].isin([1.0, 2.0, 3.0])]
        self.data['label'] = self.data['diagnosis'].astype(int) - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = np.load(row['npy_path']).astype(np.float32)

        if self.transform:
            img = self.transform(img)
            
        # already normalized in preprocessing, just add channel
        img = np.expand_dims(img, 0)  # [1,D,H,W]

        return torch.from_numpy(img), torch.tensor(row['label'])
    
    def labels(self):
        return self.data['label'].astype(int).tolist()