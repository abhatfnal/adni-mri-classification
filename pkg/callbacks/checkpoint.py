import torch
from .callback import Callback

class CheckpointManager(Callback):
    def __init__(self, monitor="val/loss", mode="min", patience=10, priority=10):
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.reset()

        self.priority = priority

    def reset(self):
        self.patience_counter = 0
        self.best_metric = None

    def _is_better(self, current, best):
        return (current < best) if self.mode == "min" else (current > best)

    def on_val_epoch_end(self, ctx):
        
        if self.monitor not in ctx["metrics"]:
            raise KeyError(f"CheckpointManager requires '{self.monitor}' in ctx.metrics")

        current = float(ctx["metrics"][self.monitor])

        if self.best_metric is None or self._is_better(current, self.best_metric):
            self.best_metric = current
            self.patience_counter = 0
            torch.save(ctx.model.state_dict(), self.save_path)
        else:
            self.patience_counter += 1

        if self.patience > 0 and self.patience_counter >= self.patience:
            ctx.signals.should_stop = True
