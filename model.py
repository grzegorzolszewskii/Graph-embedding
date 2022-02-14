from torch.nn import Module
from torch.nn import Embedding
import torch.nn.functional as fun
from torch import zeros


class Model(Module):
    def __init__(self, manifold, n, dim, sparse=False):
        super().__init__()
        self.manifold = manifold
        self.n = n
        self.dim = dim
        self.model = Embedding(n, dim, sparse=sparse)

    def forward(self, inputs):
        e = self.model(inputs)  # macierz 10x52xdim, to bedzie e to preds z pliku embed
        o = e.narrow(1, 1, e.size(1) - 1)  # wierzcholki bez 1 kolumny
        s = e.narrow(1, 0, 1).expand_as(o)
        dist = self.manifold.distance(s, o)
        return dist.squeeze(-1)

    def loss(self, inp):
        return fun.cross_entropy(inp.neg(), target=zeros())
