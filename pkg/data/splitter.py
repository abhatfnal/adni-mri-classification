import numpy as np 
from sklearn.model_selection import StratifiedGroupKFold

class Splitter:
    """ Utility class to perform stratified group k fold splitting of test, 
        train and validation indices according to specified seed, modality 
        (k-fold or holdout), indices, labels and groups. 
    """
        
    def __init__(self, indices, labels, groups, test_size=0.1, mode="kfold",
                 seed=42, folds=5, val_size=0.2):
        self.indices = np.asarray(indices)
        self.labels  = np.asarray(labels)
        self.groups  = np.asarray(groups)

        assert len(self.indices) == len(self.labels) == len(self.groups)

        self.mode = mode
        self.seed = seed
        self.test_size = test_size
        self.val_size = val_size
        self.folds = folds

        self._test_idx = None
        self._trainval_idx = None

    def test_split(self):
        if self._test_idx is not None:
            return self._test_idx

        n = len(self.indices)
        n_splits = int(np.ceil(1 / self.test_size))

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        # Work in *positions* 0..n-1
        positions = np.arange(n)
        for trainval_pos, test_pos in sgkf.split(positions, self.labels, self.groups):
            self._trainval_idx = self.indices[trainval_pos]
            self._test_idx = self.indices[test_pos]
            return self._test_idx

    def cv_split(self):
        if self._trainval_idx is None:
            self.test_split()

        # positions within trainval
        tv_mask = np.isin(self.indices, self._trainval_idx)
        tv_pos = np.where(tv_mask)[0]

        # choose n_splits
        n_splits = self.folds if self.mode == "kfold" else int(np.ceil(1 / self.val_size))

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed + 1)

        splits = []
        for train_pos, val_pos in sgkf.split(tv_pos, self.labels[tv_pos], self.groups[tv_pos]):
            train_idx = self.indices[tv_pos[train_pos]]
            val_idx   = self.indices[tv_pos[val_pos]]
            splits.append((train_idx, val_idx))
            if self.mode == "holdout":
                break

        return splits
