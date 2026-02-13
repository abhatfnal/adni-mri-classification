
class Metric:
    """
    Base class for metric
    """

    def __init__(self):
        pass 

    def reset(self):
        raise NotImplementedError
    
    def update(self, step_out: dict):
        raise NotImplementedError
    
    def compute(self):
        raise NotImplementedError

    def get_name():
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError