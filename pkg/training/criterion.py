import numpy as np
import torch
import torch.nn as nn

def build_criterion(cfg, train_labels=None, device="cpu"):

    if not "name" in cfg:
        raise KeyError("Criterion name not specified")
    
    name = cfg["name"]

    if name == "CrossEntropyLoss":

        if "params" in cfg and "weights" in cfg["params"]:
            
            if cfg["weights"] == "auto":

                if train_labels is None:
                    raise ValueError("WeightedCrossEntropyLoss requires train_labels.")
                
                counts = np.bincount(train_labels)
                weights = 1.0 / counts
                w = torch.tensor(weights, dtype=torch.float32, device=device)
                
                return nn.CrossEntropyLoss(weight=w, **cfg["params"])
        else:
            return nn.CrossEntropyLoss(**cfg["params"])
  

    elif name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(**cfg["params"])

    else:
        raise KeyError(f"Unknown criterion '{name}'")
