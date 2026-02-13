import torch
from metric import Metric

class ScalarMetric(Metric):
    """
    Average over batches a scalar metric.
    """
    def __init__(self, key: str):
        self.key = key
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, values: dict):
        if self.key not in values:
            raise KeyError(f" Missing key {self.key} for mean scalar metric update")

        val = values[self.key]

        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().item()

        self.sum += float(val)
        self.count += 1

    def compute(self):
        if self.count == 0:
            return None
        return self.sum / self.count

    def get_name(self):
        return self.key
