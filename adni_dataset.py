# adni_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torchio as tio


class ADNIDataset(Dataset):
    def __init__(self, csv_file, augment=False):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['diagnosis'].isin([1.0,2.0,3.0])]
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
        img = np.load(row['npy_path']).astype(np.float32)  # [D,H,W]

        # wrap as TorchIO subject so we can apply 3D transforms
        if self.transforms:
            subject = tio.Subject(
                mri=tio.ScalarImage(tensor=img[None]))
            subject = self.transforms(subject)
            img = subject.mri.data.squeeze(0).numpy()

        # already normalized in preprocessing, just add channel
        img = np.expand_dims(img, 0)  # [1,D,H,W]
        return torch.from_numpy(img), torch.tensor(row['label'])
