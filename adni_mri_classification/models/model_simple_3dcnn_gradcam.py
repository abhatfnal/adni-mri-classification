import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model_simple_3dcnn import Simple3DCNN

class GradCAM3DCNN(Simple3DCNN):
    
    def _build(self):
        super()._build()
        
        self.feature_maps = None

        def hook_fn(module, input, output):
            self.feature_maps = output
            if self.feature_maps.requires_grad:
                self.feature_maps.retain_grad()

        self.conv3.register_forward_hook(hook_fn)

