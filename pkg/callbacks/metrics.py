from .callback import Callback
from pkg.utils.plot import plot_metrics

import os

class Metrics(Callback):
    """
    Collects and aggregates (averages) scalar metrics from training and validation batch steps. 
    """
    def __init__(self, priority=0):

        self.current_train_metrics = {}
        self.history_train_metrics = {}

        self.current_val_metrics = {}
        self.history_val_metrics = {}

        self.batch_count = 0

        # Set lowest priority by default
        self.priority = priority

    def on_fit_start(self, context):
        
        # Initialize path 
        self.log_csv = os.path.join(context["dir"], "metrics.csv")
        self.first_epoch = True
    
    def on_train_epoch_start(self, context):
        self.batch_count = 0
        self.current_train_metrics = {}

    def on_train_batch_end(self, context, out, batch, batch_idx):
        self.batch_count += 1
        for k, v in out.items():
            if hasattr(v, "detach"):
                v = v.detach()
            self.current_train_metrics[k] = self.current_train_metrics.get(k, 0) + v

    def on_train_epoch_end(self, context):
        for k, v in self.current_train_metrics.items():
            avg = v / self.batch_count if self.batch_count > 0 else v
            context["metrics"]["train/" + k] = avg
            self.history_train_metrics.setdefault(k, []).append(avg)

    def on_val_epoch_start(self, context):
        self.batch_count = 0
        self.current_val_metrics = {}

    def on_val_batch_end(self, context, out, batch, batch_idx):
        self.batch_count += 1
        for k, v in out.items():
            if hasattr(v, "detach"):
                v = v.detach()
            self.current_val_metrics[k] = self.current_val_metrics.get(k, 0) + v

    def on_val_epoch_end(self, context):
        for k, v in self.current_val_metrics.items():
            avg = v / self.batch_count if self.batch_count > 0 else v
            context["metrics"]["val/" + k] = avg
            self.history_val_metrics.setdefault(k, []).append(avg)

        # write CSV
        keys = [k for k, _ in sorted(context["metrics"].items())]
        vals = [context["metrics"][k] for k in keys]

        # convert tensors to scalars
        def to_scalar(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu()
                if x.numel() == 1:
                    return float(x)
            return x

        vals = [to_scalar(v) for v in vals]

        with open(self.log_csv, "a") as f:
            if self.first_epoch:
                self.first_epoch = False
                f.write(",".join(keys) + "\n")
            f.write(",".join(map(str, vals)) + "\n")

    def on_fit_end(self, context):
        all_metrics = { "train/"+k: v for k, v in self.history_train_metrics.items() }
        for k, v in self.history_val_metrics.items():
            all_metrics["val/" + k] = v
        plot_metrics(os.path.join(context["dir"], "metric_plots.png"), all_metrics)
