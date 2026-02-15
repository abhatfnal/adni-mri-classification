
import torch 

class CallbacksList:
    def __init__(self, callbacks):
        self.callbacks = callbacks or []

        # Sort callbacks by ascending priority, reflecting order of execution
        self.callbacks.sort(key= lambda c : int(c.priority))

    def _call(self, name, *args, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, name, None)
            if fn is None:
                continue

    def on_fit_start(self, context):
        self._call("on_fit_start", context)

    def on_train_epoch_start(self, context):
        self._call("on_train_epoch_start", context)

    def on_train_batch_start(self, context):
        self._call("on_train_batch_start", context)

    def on_train_batch_end(self, context, out, batch,  batch_idx):
        self._call("on_train_batch_end", context, out, batch, batch_idx)

    def on_train_epoch_end(self, context):
        self._call("on_train_epoch_end", context)

    def on_fit_end(self, context):
        self._call("on_fit_end", context)

    def on_val_epoch_start(self, context):
        self._call("on_val_epoch_start", context)

    def on_val_batch_start(self, context):
        self._call("on_val_batch_start", context)

    def on_val_batch_end(self, context, out, batch,  batch_idx):
        self._call("on_val_batch_end", context, out, batch,  batch_idx)

    def on_val_epoch_end(self, context):
        self._call("on_val_epoch_end", context)


class Trainer:

    def __init__(self, task, datamodule, callbacks, max_epochs=100):

        self.task = task
        self.datamodule = datamodule
        self.max_epochs = max_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create context
        self.ctx = { "model":task.model,
                     "optimizer":task.optimizer,
                         "signals":{ 
                             "early_stop":False
                        } 
                    }

        # Create callbacks list
        self.cb = CallbacksList(callbacks)

    def fit(self):

        self.cb.on_fit_start(self.ctx)

        for epoch in range(self.epochs):

            # Stop if early stop is signalled
            if self.ctx["signals"]["early_stop"]:
                break 

            # Train 
            self.cb.on_train_epoch_start(self.ctx)
            self.train_epoch(epoch)
            self.cb.on_train_epoch_end(self.ctx)

            # Validation
            self.cb.on_val_epoch_start(self.ctx)
            self.validate_epoch(epoch)
            self.cb.on_val_epoch_end(self.ctx)
            
        self.cb.on_fit_end(self.ctx)

    def train_epoch(self, epoch):
        
        train_loader = self.datamodule.train_dataloader(epoch)

        for i, batch in enumerate(train_loader):

            # Move tensors to device
            batch = (batch[0].to(self.device), batch[1].to(self.device))

            self.cb.on_train_batch_start(self.ctx)
            out = self.task.train_batch(batch)
            self.cb.on_train_batch_end(self.ctx, out, batch, i)

    def validate_epoch(self, epoch):

        val_loader = self.datamodule.val_dataloader(epoch)

        for i, batch in enumerate(val_loader):

            # Move tensors to device
            batch = (batch[0].to(self.device), batch[1].to(self.device))
            
            self.cb.on_val_batch_start(self.ctx)
            out = self.task.validate_batch(batch)
            self.cb.on_val_batch_end(self.ctx, out, batch, i)

