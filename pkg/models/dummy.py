from .base import BaseModel
import torch
import torch.nn as nn

class Dummy(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1)) 
        pass 

    def forward(self, X):
        pass 

    def train_batch(self, batch, batch_index):
        return torch.zeros((), requires_grad=True), {"loss":0.5}

    def validate_batch(self, batch, batch_index):
        return {"loss":0.1}

    def test_batch(self, batch, batch_index):
        return {"preds":torch.tensor([1,2,1,2], requires_grad=False), "targets":torch.tensor([1,2,1,2], requires_grad=False)}
    