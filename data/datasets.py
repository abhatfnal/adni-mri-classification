import torch
import pandas as pd
import nibabel as nib
import numpy as np
import pandas as pd
import torchio as tio


from torch.utils.data import Dataset

class ADNIDataset(Dataset):
    """
    ADNI dataset class
    """
        
    def __init__(self, csv_file, transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Keep only relevant diagnoses and create labels
        self.data = self.data[self.data['diagnosis'].isin([1.0, 2.0, 3.0])]
        self.data['label'] = self.data['diagnosis'].astype(int) - 1
        self.augment = augment

        # define a 3D augmentation pipeline
        if self.augment:
            self.transforms = tio.Compose([
                tio.RandomFlip(axes=(0,1,2), p=0.5),
                tio.RandomAffine(scales=(0.9,1.1), degrees=10, p=0.5),
                tio.RandomNoise(mean=0.0, std=(0,0.1), p=0.5),
            ])
        else:
            self.transforms = None

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