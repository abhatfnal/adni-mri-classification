
class BaseTask():
    """
    Base class for Task.
    """
    def __init__(self, model, criterion, optimizer):
        pass
    
    def train_batch(self, batch):
        pass
    
    def eval_batch(self, batch):
        pass
    
    def test_batch(self, batch):
        pass 
    

class BaseClassificationTask(BaseTask):
    """
    Base classification task.
    """
    
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optim = optimizer
    
    def train_batch(self, batch):
        
        # Get features and labels
        X, y = batch 
        
        # Zero grad 
        self.optim.zero_grad()
        
        # Forward
        loss = self.criterion( self.model(X), y)
        
        # Backwards pass 
        loss.backward()
        
        # Optimizer step
        self.optim.step()
        
        return loss.item()
    
    def eval_batch(self, batch):
        pass
    
    def test_batch(self, batch):
        pass 
    