# adni_dataset.py

import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np

class ADNIDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Use numeric labels from the CSV (already 1.0, 2.0, 3.0)
        self.data = self.data[self.data['diagnosis'].isin([1.0, 2.0, 3.0])]
        self.data['label'] = self.data['diagnosis'].astype(int) - 1  # Map 1→0 (CN), 2→1 (MCI), 3→2 (AD)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = nib.load(row['filepath']).get_fdata().astype(np.float32)

        # Normalize the image
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        img = np.expand_dims(img, axis=0)  # Add channel dim

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img), torch.tensor(row['label'])
