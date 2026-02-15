from .task import Task

class DummyTask(Task):

    def __init__(self):
        self.count = 0
        pass
    
    def setup(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        print("Dummy task init")
    
    def train_batch(self, batch):
        print("Train batch!")
        return {"acc":0.1*self.count**2, "loss":(1 - 0.1*self.count) }
    
    def validate_batch(self, batch):
        print("Validate batch")
        return {"acc":0.45*self.count**1.5, "loss":(2 - 0.2*self.count**2) }