from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR

SCHEDULER_REGISTRY = {
    "ReduceLROnPlateau": {"class": ReduceLROnPlateau, "step_on": "epoch", "monitor": "val/loss"},
    "CosineAnnealingLR": {"class": CosineAnnealingLR, "step_on": "epoch", "monitor": None},
    "OneCycleLR": {"class": OneCycleLR, "step_on": "batch", "monitor": None},
}

class SchedulerPolicy:

    def __init__(self, name, params, optimizer):
        
        if name not in SCHEDULER_REGISTRY:
            raise ValueError(f"Unknown scheduler '{name}'. Options: {list(SCHEDULER_REGISTRY)}")

        self.conf = SCHEDULER_REGISTRY[name]
        self.name = name
        self.scheduler = self.conf["class"](optimizer, **params)

    def on_train_batch_end(self):
        if self.conf["step_on"] == "batch":
            self.scheduler.step()

    def on_epoch_end(self, metrics: dict | None = None):
        if self.conf["step_on"] != "epoch":
            return

        monitor_key = self.conf["monitor"]
        if monitor_key is None:
            self.scheduler.step()
        else:
            if metrics is None or monitor_key not in metrics:
                raise KeyError(f"Scheduler '{self.name}' requires metric '{monitor_key}' at epoch end.")
            self.scheduler.step(metrics[monitor_key])
