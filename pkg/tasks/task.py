
class Task():
    """
    Base task class. Handles training, evaluation and testing of the model
    on single batches of data.
    """
    def __init__(self):
        raise NotImplementedError
    
    def setup(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
    
    def train_batch(self, batch):
        raise NotImplementedError
    
    def validate_batch(self, batch):
        raise NotImplementedError
    
    def test_batch(self, batch):
        return self.validate_batch(batch)

    def get_optimizer(self):
        return self.optim
    
    def get_model(self):
        return self.model
