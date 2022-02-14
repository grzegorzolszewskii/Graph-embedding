from graph_import import load_graph
import torch as th
import numpy as np
from random import choice, randint
from model import Model
from manifolds import Euclidean


def train(vertices_num, model=None, optimizer=None):
    graph = load_graph(vertices_num)
    epoch_loss = th.Tensor(len(graph))  # robi losowy wektor dlugosci len(graph)

    # chcemy zeby ilosc wierzch. byla podzielna przez 10
    if len(graph) % 10 != 0:
        raise ValueError("Liczba wierzcholkow musi byc podzielna przez 10")

    for epoch in range(0, int(len(graph)/10)):
        epoch_loss.fill_(0)
        inputs = th.zeros(size=(10, 52))
        for i in range(0, 10):  # tworzenie macierzy 10x52
            batch = 10*epoch+i+1
            batch_connected = choice(tuple(graph[batch]))
            inputs[i, 0] = batch
            inputs[i, 1] = batch_connected
            for j in range(2, 52):  # jest 50 losowan wiec duza szansa ze wylosuje sie polaczony
                inputs[i, j] = randint(1, vertices_num)
                while inputs[i, j].item() in graph[batch]:
                    inputs[i, j] = randint(1, vertices_num)
                    # czy niepolaczone moga sie powtarzac w wierszu? chyba tak

    return inputs


if __name__ == '__main__':
    print(train(330))

