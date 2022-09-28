from torch.nn import Module
from torch.nn import Embedding
import torch.nn.functional as fun
import torch as th


th.set_default_tensor_type('torch.DoubleTensor')


class Model(Module):
    def __init__(self, manifold, n, dim, alpha=1, sparse=False):
        super().__init__()
        self.manifold = manifold
        self.n = n
        self.dim = dim
        self.alpha = alpha
        self.model = Embedding(n, dim, sparse=sparse)
        if self.manifold.manifold_type == 'lorentz':
            self.manifold.init_weights(self.model)

    def forward(self, inputs):                  # inputs to preds - matrix 10x52 to 10x52xdim
        e = self.model(inputs)                  # e is 10x52xdim matrix, it is preds from embed.py - Embedding here
        o = e.narrow(1, 1, e.size(1) - 1)       # e without first column (10x51xdim)
        s = e.narrow(1, 0, 1).expand_as(o)      # 10x51xdim matrix - row is repeated first column of e
        dist = self.manifold.distance(s, o)
        return dist.squeeze(-1)

    def loss(self, inp, target, **kwargs):
        return fun.cross_entropy(-self.alpha*inp, target)

    def optim_params(self):
        return [{
            'params': self.model.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]
