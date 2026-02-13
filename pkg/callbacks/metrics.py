from .callback import Callback

class Metrics(Callback):
    """
    Collects and aggregates (averages) scalar metrics from training and validation batch steps. 
    """

    def __init__(self):
        pass

    def on_train_epoch_start(self, context):
        pass
        
    def on_train_batch_end(self, context, out, batch, batch_idx):
        pass
    def on_train_epoch_end(self, context):
        pass

    def on_val_epoch_start(self, context):
        pass

    def on_val_batch_end(self, context, out, batch, batch_idx):
        pass

    def on_val_epoch_end(self, context):
        pass