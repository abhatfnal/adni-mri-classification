import torch
import torch.nn as nn


# Base class for models
class BaseModel(nn.Module):

    def set_criterion(self, criterion):
        """
        Sets criterion to be used.
        """
        self.criterion = criterion

    def train_batch(self, batch, batch_index) -> tuple[torch.Tensor, dict]:
        """
        Trains model on a batch. Returns the loss tensor
        and a dict with metrics.
        """
        pass 

    def validate_batch(self, batch, batch_index) -> dict:
        """
        Validates the model on a batch. Returns dict with metrics.
        """
        pass

    def test_batch(self, batch, batch_index) -> dict:
        return self.validate_batch(batch, batch_index)