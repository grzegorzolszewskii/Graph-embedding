import torch as th
from math import pi
import pandas as pd
import csv


def vectors():
    e = th.zeros(size=(10, 52))
    for i in range(0, 10):
        for j in range(0, 52):
            if j == 0:
                e[i, j] = 99 + i
            else:
                e[i, j] = i + j

    # print("Wektor e: ", e)  # macierz 10x52xdim, e to preds z pliku embed - tu wykonuje sie zanurzenie
    o = e.narrow(1, 1, e.size(1) - 1)  # wierzcholki bez 1 kolumny
    s = e.narrow(1, 0, 1).expand_as(o)

    # print("Wektor o: ", o)  # macierz e bez 1 kolumny (10x51xdim)
    # print("Wektor s: ", s)  # macierz 10x51xdim - kolumny to powt. sie 1 kolumna z e
    # print(len(s[0]))
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
                    e[i, j, k] = i*j*k*pi

    # print("Wektor e: ", e)
    o = e.narrow(1, 1, e.size(1) - 1)  # wierzcholki bez 1 kolumny
    s = e.narrow(1, 0, 1).expand_as(o)

    # print("Wektor o: ", o)
    # print("Wektor s: ", s)
    # print(len(s[0]))
    return e, o, s


def distance(u, v):
    return (u - v).pow(2)


dist = distance(moredim_vectors()[1], moredim_vectors()[2])
# print("dist: ", dist)
# print("dist sum dim -1: ", dist.sum(dim=-1))

df = pd.read_csv('best_embedding', header=None)
for i in range(4):
    print(df[i][0])
print({i: [0 for j in range(4+1)] for i in [1,2,3,4,5]})

