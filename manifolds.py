import torch as th
import numpy as np
from numpy import arccosh, cosh, sinh


class Manifold:
    def __init__(self, manifold_type, max_norm=None, K=None, **kwargs):
        self.manifold_type = manifold_type
        self.max_norm = max_norm
        self.K = K
        if K is not None:
            self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))

    def init_weights(self, w, scale=1e-4):
        w.weight.data.uniform_(-scale, scale)

    def distance(self, u, v):
        if self.manifold_type == 'euclidean':
            return ((u - v).pow(2)).sum(dim=-1)
        if self.manifold_type == 'lorentz':
            x1 = u[:, :, 0]
            y1 = u[:, :, 1]
            x2 = v[:, :, 0]
            y2 = v[:, :, 1]
            return arccosh(cosh(y1)*cosh(x1-x2)*cosh(y2) - sinh(y1)*sinh(y2))
