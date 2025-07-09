import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from .settings import DATA_DIR

CSV_FILE = os.path.join(DATA_DIR, 'adni_preprocessed_npy_metadata.csv')

class ADNIDataset(Dataset):
    """
    ADNI dataset class
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
        
        full_path = row['npy_path'] # Paths are by construction absolute

        img = np.load(full_path).astype(np.float32)
        
        # Already normalized: add channel dimension
        img = np.expand_dims(img, axis=0)

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img), torch.tensor(row['label'])


class TrainValDataset(Dataset):
    """
    Dataset for training and validation, created with trainval_indices.npy
    """
    def __init__(self, transform=None):
        # Load metadata
        self.data = pd.read_csv(CSV_FILE)
        self.transform = transform

        # Filter relevant diagnoses and create labels
        self.data = self.data[self.data['diagnosis'].isin([1.0, 2.0, 3.0])]
        self.data['label'] = self.data['diagnosis'].astype(int) - 1

        # Load train+val split indices
        indices_path = os.path.join(DATA_DIR, 'trainval_indices.npy')
        self.indices = np.load(indices_path).astype(int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map to original row index
        data_idx = int(self.indices[idx])
        row = self.data.iloc[data_idx]

        full_path = row['npy_path'] # Paths are by default absolute

        img = np.load(full_path).astype(np.float32)
        img = np.expand_dims(img, axis=0)

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img), torch.tensor(row['label'])
    
    def labels(self):
        return [ self.data.iloc[ int(self.indices[i]) ]['label'] for i in range(len(self)) ]
    


class TestDataset(Dataset):
    """
    Dataset for testing / evaluation, created with test_indices.npy
    """
    def __init__(self, transform=None):
        # Load metadata
        self.data = pd.read_csv(CSV_FILE)
        self.transform = transform

        # Filter relevant diagnoses and create labels
        self.data = self.data[self.data['diagnosis'].isin([1.0, 2.0, 3.0])]
        self.data['label'] = self.data['diagnosis'].astype(int) - 1

        # Load test split indices
        indices_path = os.path.join(DATA_DIR, 'test_indices.npy')
        self.indices = np.load(indices_path).astype(int)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map to original row index
        data_idx = int(self.indices[idx])
        row = self.data.iloc[data_idx]

        full_path = row['npy_path'] # Paths are bu default absolute

        img = np.load(full_path).astype(np.float32)
        img = np.expand_dims(img, axis=0)

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img), torch.tensor(row['label'])
    
    def labels(self):
        return [ self.data.iloc[ int(self.indices[i]) ]['label'] for i in range(len(self)) ]
