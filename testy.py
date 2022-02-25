import torch as th
from math import pi


def vectors():
    e = th.zeros(size=(10, 52))
    for i in range(0, 10):
        for j in range(0, 52):
            if j == 0:
                e[i, j] = 99 + i
            else:
                e[i, j] = i + j

    print("Wektor e: ", e)
    o = e.narrow(1, 1, e.size(1) - 1)  # wierzcholki bez 1 kolumny
    s = e.narrow(1, 0, 1).expand_as(o)

    print("Wektor o: ", o)
    print("Wektor s: ", s)
    print(len(s[0]))
    return e, o, s


def moredim_vectors(dim=2):
    e = th.zeros(size=(10, 52, dim))
    for i in range(0, 10):
        for j in range(0, 52):
            for k in range(0, dim):
                if j == 0 and k == 0:
                    e[i, j, k] = 99 + i
                if j == 1 and k == 0:
                    e[i, j, k] = 299 + i
                if j != 0 and j != 1 and k == 0:
                    e[i, j, k] = i + j
                if k != 0:
                    e[i, j, k] = pi

    print("Wektor e: ", e)
    o = e.narrow(1, 1, e.size(1) - 1)  # wierzcholki bez 1 kolumny
    s = e.narrow(1, 0, 1).expand_as(o)

    print("Wektor o: ", o)
    print("Wektor s: ", s)
    print(len(s[0]))
    return e, o, s


def distance(u, v):
    return ((u - v).pow(2)).pow(1 / 2).sum(dim=-1)


dist = distance(moredim_vectors()[1], moredim_vectors()[2])
print(dist.squeeze(-1))
