from graph_import import load_graph
import torch as th
from random import choice, randint, seed

seed(1)


def train(nodes_num, model, optimizer, epochs=50):
    graph = load_graph(nodes_num)

    if len(graph) % 10 != 0:
        raise ValueError("Liczba wierzcholkow musi byc podzielna przez 10")

    loss_list = [0 for i in range(epochs)]
    for epoch in range(epochs):
        for batch in range(0, int(len(graph) / 10)):

            inputs = th.randint(nodes_num, (10, 52))
            for i in range(0, 10):
                node = 10 * batch + i + 1
                node_connected = choice(tuple(graph[node]))
                inputs[i, 0] = node
                inputs[i, 1] = node_connected
                for j in range(2, 52):  # jest 50 losowan wiec duza szansa ze wylosuje sie polaczony
                    inputs[i, j] = randint(1, nodes_num)
                    while inputs[i, j].item() in graph[node]:
                        inputs[i, j] = randint(1, nodes_num)

            optimizer.zero_grad()  # za kazdym razem chcemy nowy gradient
            preds = model(inputs)[0]
            target = th.zeros(10).long()

            # zapisuje wierzcholki i ich wspolrzedne w ostatniej epoce zeby je narysowac
            if epoch == epochs - 1 and batch == int(len(graph) / 10) - 1:
                nodes = inputs
                coordinates = model(inputs)[1]
                print("Jestem w zapisie, numery epoki i batcha to: ", epoch, batch)

            loss = model.loss(preds, target=target, size_average=True)
            # print(loss.item())
            loss_list[epoch] += loss.item()

            loss.backward()  # obliczenie pochodnych
            optimizer.step()  # dodaj pochodne do pozycji wierzch

    return loss_list, nodes, coordinates
