import torch as th
import random as rand
from graph_import import load_graph
from manifolds import Manifold
from model import Model
from rsgd import RiemannianSGD


def train(graph, model, optimizer, epochs=50, max_loss=2.5, m_rows=10, m_cols=52):
    nodes_num = len(graph)
    loss_list = [0 for i in range(epochs)]

    for epoch in range(epochs):
        for batch in range(0, int(len(graph) // m_rows)):
            inputs = th.randint(nodes_num, (m_rows, m_cols))    # th.randint bierze z [0, a)
            for i in range(0, m_rows):                      # random.randint bierze z [a, b]
                node = m_rows * batch + i
                node_connected = rand.choice(tuple(graph[node]))
                inputs[i, 0] = node
                inputs[i, 1] = node_connected
                for j in range(2, m_cols):      # jest 50 losowan - duza szansa ze wylosuje sie polaczony
                    inputs[i, j] = rand.randint(0, nodes_num-1)
                    while inputs[i, j].item() in graph[node]:
                        inputs[i, j] = rand.randint(0, nodes_num-1)

            if epoch < 50:
                optimizer.lr = 0.001

            optimizer.zero_grad()       # za kazdym razem chcemy nowy gradient
            preds = model(inputs)
            target = th.zeros(m_rows).long()

            loss = model.loss(preds, target=target, size_average=True)
            loss_list[epoch] += loss.item()

            loss.backward()  # obliczenie pochodnych
            optimizer.step()  # dodaj pochodne do pozycji wierzch

        if epoch > epochs/2:
            if loss_list[epoch] < min(loss_list[:epoch]):
                if loss_list[epoch] <= max_loss:
                    print("Przerywam zanurzanie dla loss rownego: ", loss_list[epoch])
                    return loss_list[epoch], model.model.weight

    return loss_list[-1], model.model.weight
