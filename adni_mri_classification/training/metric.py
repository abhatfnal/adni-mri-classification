class BaseMetric:
    def reset(self):
        raise NotImplementedError
    def update(self, step_out: dict):
        raise NotImplementedError
    def compute(self):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError


class LossMetric(BaseMetric):
    """
    Averages one or more loss scalars over many updates.
    Expects update() to receive a dict like {"loss": 0.7, "loss_aux": 0.2}
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {}     # key -> float
        self.counts = {}   # key -> int

    def update(self, values: dict):
        for k, v in values.items():
            # ensure python float
            v = float(v)
            self.sums[k] = self.sums.get(k, 0.0) + v
            self.counts[k] = self.counts.get(k, 0) + 1

    def compute(self):
        if not self.sums:
            return None
        return {k: self.sums[k] / self.counts[k] for k in self.sums}
