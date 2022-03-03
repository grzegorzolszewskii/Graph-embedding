from graph_import import load_graph
import torch as th
import numpy as np
from random import choice, randint
from model import Model
from manifolds import Euclidean
from torch.optim.optimizer import Optimizer

eucl = Euclidean()
model_n520 = Model(eucl, 520, 2)
model_n331 = Model(eucl, 331, 2)
optimizer = th.optim.SGD(model_n331.parameters(), lr=0.1)


def train(vertices_num, model=None, optimizer=None):
    graph = load_graph(vertices_num)
    epoch_loss = th.Tensor(len(graph))  # robi losowy wektor dlugosci len(graph)

    # chcemy zeby ilosc wierzch. byla podzielna przez 10
    if len(graph) % 10 != 0:
        raise ValueError("Liczba wierzcholkow musi byc podzielna przez 10")

    loss_list = [0 for i in range(0, 100)]
    for epoch in range(0, 10):
        for mini_batch_index in range(0, int(len(graph)/10)):
            epoch_loss.fill_(0)
            inputs = th.randint(vertices_num, (10, 52))  # 10 wierszy, 52 kolumny
            for i in range(0, 10):  # tworzenie macierzy 10x52
                batch = 10*mini_batch_index+i+1
                batch_connected = choice(tuple(graph[batch]))
                inputs[i, 0] = batch
                inputs[i, 1] = batch_connected
                for j in range(2, 52):  # jest 50 losowan wiec duza szansa ze wylosuje sie polaczony
                    inputs[i, j] = randint(1, vertices_num)
                    while inputs[i, j].item() in graph[batch]:
                        inputs[i, j] = randint(1, vertices_num)
                        # czy niepolaczone moga sie powtarzac w wierszu? chyba tak

            optimizer.zero_grad()  # za kazdym razem chcemy nowy gradient
            preds = model_n331(inputs)
            print(preds)
            target = th.zeros(10).long()

            loss = model_n331.loss(preds, target=target, size_average=True)
            print(loss.item())
            loss_list[epoch] += loss.item()

            loss.backward()  # obliczenie pochodnych
            optimizer.step()  # dodaj pochodne do pozycji wierzch

    return loss_list


if __name__ == '__main__':
    print(train(330, model_n331, optimizer=optimizer))

