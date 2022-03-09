from torch.nn import Module
from torch.nn import Embedding
import torch.nn.functional as fun
import torch as th


class Model(Module):
    def __init__(self, manifold, n, dim, sparse=False):
        super().__init__()
        self.manifold = manifold
        self.n = n
        self.dim = dim
        self.model = Embedding(n, dim, sparse=sparse)

    def forward(self, inputs):  # z inputs zrobi preds - z macierzy 10x52 zrobi sie 10x52xdim
        e = self.model(inputs)  # macierz 10x52xdim, e to preds z pliku embed - tu wykonuje sie zanurzenie
        o = e.narrow(1, 1, e.size(1) - 1)   # macierz e bez 1 kolumny (10x51xdim)
        s = e.narrow(1, 0, 1).expand_as(o)  # macierz 10x51xdim - wiersz to powt. sie 1 kolumna z e
        dist = self.manifold.distance(s, o)
        return dist.squeeze(-1), e

    def loss(self, inp, target, **kwargs):
        return fun.cross_entropy(inp.neg(), target)
