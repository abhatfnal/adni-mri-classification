import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DummyModel(nn.Module):

    def __init__(self, **params):
        pass 

    def forward(self, x):
        pass 