import torch
import torch.nn as nn

from abc import ABC, abstractmethod

# Base class for models
class BaseModel(nn.Module, ABC):
    
    def __init__(self, cfg: dict):
        super().__init__()            # initializes nn.Module
        self.cfg = cfg                # store hyperparams, architecture params, etc.
        self._build()           # subclass must define this

    @abstractmethod
    def _build(self):
        """
        Read self.cfg to:
          - construct layers
          - assign them to self (e.g. self.net = nn.Sequential(...))
        """
        pass

    @abstractmethod
    def forward(self, x):
        """Define the forward pass. Must return model output."""
        pass