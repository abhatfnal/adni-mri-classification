"""
Splits dataset into train+val and test indices, saving indices to files.
To be run once at the very beginning of experimentation, after preprocessing.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from settings import SPLIT_RANDOM_STATE, DATA_DIR

csv_file = os.path.join(DATA_DIR,'adni_preprocessed_npy_metadata.csv')

df = pd.read_csv(csv_file)
labels = df['diagnosis'].astype(int).tolist()

# 90% training+validation, 10% test
trainval_idx, test_idx = train_test_split(
    np.array( list(range(len(df))) ),
    test_size=0.1,
    stratify=labels,
    random_state=SPLIT_RANDOM_STATE
)

# save indices to file for loader use
np.save('trainval_indices.npy',trainval_idx)
np.save('test_indices.npy',test_idx)




