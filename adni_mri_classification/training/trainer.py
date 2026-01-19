

class Trainer:
    """
    Model trainer.
    """
    
    def __init__(self, 
                 model,
                 device,
                 task,
                 scheduler,
                 augmentation,
                 train_metric, 
                 val_metric):
        
        self.model  = model
        self.device = device
        self.task = task
        self.scheduler = scheduler
        self.augmentation = augmentation
        self.train_metric = train_metric
        self.val_metric = val_metric
        
    def train_epoch(self, epoch, train_loader, val_loader):
        
        # Reset metrics
        self.train_metric.reset()
        self.val_metric.reset()
        
        # Set model to training mode
        self.model.train()
        
        # Loop over batches
        for X,y in train_loader:
            
            # Move tensors to device
            batch = (X.to(self.device), y.to(self.device))
            
            # Apply inside-batch augmentation
            batch = self.augmentation.transform_batch(batch)
            
            # Train model on one batch
            self.train_metric.update( self.train_batch((X,y)))
            
            # Update scheduler policy in-batch
            self.scheduler.batch_step()
            
        #Evaluation mode
        self.model.eval()
        
        # Evaluate model on validation dataset
        for X,y in val_loader:
            
            # Move tensors to device
            batch = (X.to(self.device), y.to(self.device))
            
            # Evaluate model on single batch
            self.val_metric.update(self.eval_batch(batch))
            
        # Update scheduler policy in-epoch
        self.scheduler.epoch_step(self.val_metric.compute())
        
    def train_batch(self, batch):

        return self.task.train_batch(batch)
        
    def eval_batch(self, batch):
        
        return self.task.eval_batch(batch)
    
    def current_metrics(self):
        
        return {"train_metric":self.train_metric.compute(),
                "val_metric":self.val_metric.compute()}
        
    