import numpy as np
import torch
from sklearn.metrics import classification_report

from metric import Metric


class ClassificationReportMetric(Metric):
    """
    Accumulates predictions and targets over batches.
    Returns classification report.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, values : dict):

        if "preds" not in values or "targets" not in values:
            raise KeyError("ConfusionMatrixMetric requires 'preds' and 'targets'")

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

        return classification_report(targets, preds)

    def get_name(self):
        return "classification_report"
