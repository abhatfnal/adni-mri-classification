from .task import Task

class ClassificationTask(Task):
    """
    Basic task class for classification.
    """
    
    def __init__(self, output_mode):
        pass        

    def setup(self, model, criterion, optimizer):
        self.model = model
        self.optimizer = optimizer
        
    
    def train_batch(self, batch):
        
        # Get features and labels
        X, y = batch 
        
        # Zero grad 
        self.optim.zero_grad()
        
        # Forward 
        y_hat = self.model(X)

        # Loss
        loss = self.criterion(y_hat, y)
        
        # Backwards pass 
        loss.backward()
        
        # Optimizer step
        self.optim.step()
        
        return {"loss":float(loss.item()), "logits":y_hat.detach(), "targets":y.detach()}
    
    def validate_batch(self, batch):
        
        # Get features and labels
        X, y = batch 
        
        # Forward + logits
        y_hat = self.model(X)

        # Loss
        loss = self.criterion( y_hat, y)
     
        return {"loss":float(loss.item()), "logits":y_hat.detach(), "targets":y.detach()}