import numpy as np

from torch.utils.data import WeightedRandomSampler


def build_sampler(name, train_labels, params):
    
    if name == "WeightedRandomSampler":

        # Compute weights
        class_weights = 1/np.bincount(train_labels)
        sample_weights = class_weights[train_labels]

        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    
    else:
        raise KeyError(f"Unknown sampler {name}")
        
