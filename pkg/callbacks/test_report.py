import torch
import os
import pandas as pd
from .callback import Callback
from sklearn.metrics import classification_report, confusion_matrix

class TestReport(Callback):
    """
    Receives predictions and targets on test batches.
    Computes classification report, confusion matrix and other test metrics.
    """
    def __init__(self, priority=10):
        self.priority = priority

    def on_test_start(self, context):

        # Load best model state 
        context["model"].load_state_dict(torch.load(context["checkpoint_path"], weights_only=True))

        # Initialize predictions and targets
        self.preds = []
        self.targets = []
        

    def on_test_batch_end(self, context, out, batch, batch_idx):
        
        preds = out["preds"]
        targets = out["targets"]

        if not isinstance(preds, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("Expected preds/targets to be torch.Tensor")

        # common shapes:
        # preds: (B,) class indices OR (B, C) logits
        # targets: (B,)
        if preds.ndim > 1:
            preds = preds.argmax(dim=-1)

        preds = preds.detach().cpu().view(-1)
        targets = targets.detach().cpu().view(-1)

        self.preds.append(preds)
        self.targets.append(targets)
        
    def on_test_end(self, context):
        
        if len(self.preds) == 0 or len(self.targets) == 0:
            raise RuntimeError(f"No data collected: preds={len(self.preds)} targets={len(self.targets)}")

        preds = torch.cat(self.preds).numpy()
        targets = torch.cat(self.targets).numpy()

        cm = pd.DataFrame(confusion_matrix(targets, preds))
        report = pd.DataFrame(classification_report(targets, preds, digits=4, output_dict=True)).T

        print(cm)
        print(report)

        # Save to .csv 
        cm.to_csv(os.path.join(context["dir"], "confusion_matrix.csv"))
        report.to_csv(os.path.join(context["dir"], "classification_report.csv"))

        # reset for next run
        self.preds.clear()
        self.targets.clear()
        