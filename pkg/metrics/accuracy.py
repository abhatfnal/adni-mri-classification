import torch
import numpy as np
from metric import Metric

class AccuracyMetric(Metric):

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, values : dict):

        if "preds" not in values or "targets" not in values:
            raise KeyError("AccuracyMetric requires 'preds' and 'targets'")

        preds = values["preds"]
        targets = values["targets"]

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.targets.append(targets)
        self.preds.append(preds)

    def compute(self):
        if len(self.preds) == 0:
            return None
        
        preds = np.concatenate(self.preds)
        targets = np.concatenate(self.targets)

        return float((preds == targets).mean())

    def get_name(self):
        return "accuracy"