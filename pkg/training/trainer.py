
class CallbacksList:
    def __init__(self, callbacks):
        self.callbacks = callbacks or []

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

    def __init__(self, task, datamodule, callbacks, epochs):

        self.task = task
        self.datamodule = datamodule
        self.epochs = epochs 

        # Create context
        self.ctx = { "model":task.model, 
                         "signals":{ 
                             "early_stop":False
                        } 
                    }

        # Create callbacks list
        self.cb = CallbacksList(callbacks)

    def fit(self):

        self.cb.on_fit_start(self.ctx)

        for epoch in range(self.epochs):

            self.cb.on_train_epoch_start(self.ctx)
            self.train_epoch(epoch)
            self.cb.on_train_epoch_end(self.ctx)

            self.cb.on_val_epoch_start(self.ctx)
            self.validate_epoch(epoch)
            self.cb.on_val_epoch_end(self.ctx)
            
        self.cb.on_fit_end(self.ctx)

    def train_epoch(self, epoch):
        
        train_loader = self.datamodule.train_dataloader()

        for i, batch in enumerate(train_loader):

            self.cb.on_train_batch_start(self.ctx)
            out = self.task.train_batch(batch)
            self.cb.on_train_batch_end(self.ctx, out, batch, i)

    def validate_epoch(self, epoch):

        val_loader = self.datamodule.val_dataloader()

        for i, batch in enumerate(val_loader):

            self.cb.on_val_batch_start(self.ctx)
            out = self.task.validate_batch(batch)
            self.cb.on_val_batch_end(self.ctx, out, batch, i)

