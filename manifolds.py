import torch as th
import numpy as np


class Euclidean:
    def __init__(self, max_norm=None, K=None, **kwargs):
        self.max_norm = max_norm
        self.K = K
        if K is not None:
            self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))

    def init_weights(self, w, scale=1e-4):
        w.weight.data.uniform_(-scale, scale)

    def distance(self, u, v):
        return ((u - v).pow(2)).sum(dim=-1)
