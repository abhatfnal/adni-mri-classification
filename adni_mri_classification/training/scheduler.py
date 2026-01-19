

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR

# Registry for schedulers
SCHEDULER_REGISTRY = {
    "ReduceLROnPlateau":{"class": ReduceLROnPlateau, "step_on":"epoch", "monitor":"val"},
    "CosineAnnealingLR":{"class":CosineAnnealingLR, "step_on":"epoch", "monitor":None},
    "OneCycleLR":{"class":OneCycleLR, "step_on":"batch", "monitor":None}
}

class Scheduler():
    """
    Handy wrapper for PyTorch schedulers
    """
    
    def __init__(self, name, params, optimizer):
        
        # Save config
        self.conf = SCHEDULER_REGISTRY[name]
        
        # Instantiate class and pass parameters
        self.scheduler = self.conf["class"](optimizer, **params)
        
    def batch_step(self):
        
        if self.conf["step_on"] == "batch":
            self.scheduler.step()
    
    def epoch_step(self, val):
        
        if self.conf["step_on"] == "epoch":
            if self.conf["monitor"] == "val":
                self.scheduler.step(val)
            self.scheduler.step()
        