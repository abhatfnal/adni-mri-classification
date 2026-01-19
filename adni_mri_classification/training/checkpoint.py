import torch

class CheckpointManager:
    """
    Basic Checkpoint manager. Tracks metrics, implements early stopping using patience, 
    tracks best states using monitored metric and mode.
    """
    
    def __init__(self, save_path, monitor="val/loss", mode="min", patience=0):
        
        self.save_path = save_path      # Model state save path
        self.monitor = monitor          # Metric to monitor
        self.mode = mode                # Monitor mode: max or min
        self.patience = int(patience)   # Patience
        
        # Initialize params
        self.reset()

    def _is_better(self, current, best):
        
        # Compares monitored metrics according to comparison mode
        return (current < best) if self.mode == "min" else (current > best)

    def update(self, epoch, metrics: dict) -> bool:
        
        # store full history
        row = {"epoch": epoch, **metrics}
        self.history.append(row)

        # Update best value of monitored metric
        current = metrics[self.monitor]
        if self.best_value is None or self._is_better(current, self.best_value):
            
            # Update checkpoint
            self.best_value = current
            self.best_epoch = epoch
            self.best_metrics = dict(metrics)
            
            # Reset patience counter
            self.patience_counter = 0
            return True

        # Increase patience counter
        self.patience_counter += 1
        return False

    def early_stop(self) -> bool:
        return self.patience > 0 and self.patience_counter >= self.patience

    def save_state(self, state_dict):
        
        # Save to specified path
        torch.save(state_dict, self.save_path)

    def get_save_path(self):
        
        # Return save path
        return self.save_path
    
    def reset(self):
        
        # Initialize parameters
        self.history = []
        self.best_value = None
        self.best_epoch = None
        self.best_metrics = None
        self.patience_counter = 0
