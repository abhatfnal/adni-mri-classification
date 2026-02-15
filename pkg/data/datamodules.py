from .datasets import ADNIDataset, TransformDataset
from .splitter import Splitter
from torch.utils.data import DataLoader, Subset
from .augmentation import build_augmentation
from .loaders import build_loader
import numpy as np 

class DataModule:
    """
    Skeleton for DataModule class.
    """
    
    def __init__(self ):
        raise NotImplementedError

    def setup(self, dataset, split, loader, transform):
        raise NotImplementedError

    def set_fold(idx):
        raise NotImplementedError

    def n_folds():
        raise NotImplementedError
    
    def train_loader(epoch):
        raise NotImplementedError 

    def val_loader(epoch):
        raise NotImplementedError

    def test_loader(epoch):
        raise NotImplementedError


class ADNIDataModule(DataModule):
    """
    Data module for the ADNI dataset. Returns train, validation and test loaders,
    given cross validation splitting, dataset, loader and augmentation transform configs. 
    """

    def __init__(self, data, split, loader, transform):

        self.data_cfg = data
        self.split_cfg = split
        self.loader_cfg = loader
        self.transform_cfg = transform

        self.fold_index = 0

        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    def setup(self):

        # Create dataset
        self.ds = ADNIDataset(**self.data_cfg)

        # Set it up
        self.ds.setup()

        # Create splitter
        splitter = Splitter(range(len(self.ds)), self.ds.labels(), self.ds.groups(),**self.split_cfg)

        # Create test dataset and loader
        test_dataset = Subset(self.ds, splitter.test_split())
        self._test_loader = build_loader(test_dataset, labels=None)

        # Get train and validation indices for each fold
        self.folds = splitter.cv_split()
        
    def set_fold(self, idx):

        if idx > len(self.folds) - 1:
            raise IndexError(f"Fold index {idx} out of bounds for {len(self.folds)} folds")

        self.fold_index = idx

        # Indices   
        train_idxs = self.folds[idx][0]
        val_idxs = self.folds[idx][1]

        # Augmentation transform
        augmentation = build_augmentation(self.transform_cfg)
        
        # Datasets
        train_ds = TransformDataset(Subset(self.ds, train_idxs), transform=augmentation)
        val_ds = Subset(self.ds, val_idxs)

        # Store train_labels (can be used for computing weights for loss criterion)
        self.train_labels = np.array(self.ds.labels())[train_idxs]

        # Loaders 
        self._train_loader = build_loader(train_ds, labels=self.train_labels, **self.loader_cfg)
        self._val_loader = build_loader(val_ds)

    def n_folds(self):
        return len(self.folds)
    
    def train_dataloader(self, epoch):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader



    