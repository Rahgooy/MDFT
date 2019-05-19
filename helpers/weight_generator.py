import numpy as np


class WeightGenerator:
    def __init__(self, w):
        self.w = w

    def generate(self):
        pass


class RouletteWheelGenerator(WeightGenerator):
    def __init__(self, w):
        super().__init__(w)

    def generate(self):
        r = np.random.uniform()
        wheel = self.w.cumsum(axis=0)
        winner = (wheel < r).sum()

        W = np.zeros(self.w.shape)
        W[winner] = 1
        return W
