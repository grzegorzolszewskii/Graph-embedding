import matplotlib.pyplot as plt
from model import Model
from manifolds import Euclidean
import torch as th
from train_function import train
from graph_import import load_graph


def connected_with_v(v, nodes, coordinates):  # argumentami sa macierze: 10x52, 10x52x2
    v_connected = []
    nodes_set = {nodes[i, j].item() for i in range(10) for j in range(52)}

    if v not in nodes_set:
        raise ValueError("Wierzcholek nie znalazl sie w ostatniej epoce")

    for i in graph[v]:
        if i in nodes_set:
            v_connected.append(i)
            # nie wszystkie nodes polaczone z 1 zalapaly sie w ostatnim epoch - bierzemy tylko te ktore sie zalapaly

    v_coords = ()
    X = []
    Y = []
    for w in v_connected:
        for i in range(10):
            for j in range(52):
                if nodes[i, j].item() == w:
                    X.append(coordinates[i, j, 0].item())
                    Y.append(coordinates[i, j, 1].item())
                if nodes[i, j].item() == v:
                    v_coords = coordinates[i, j, 0].item(), coordinates[i, j, 1].item()

    return v_connected, X, Y, v_coords


nodes_num = 330
eucl = Euclidean()
model_n331 = Model(eucl, 331, 2)
optimizer = th.optim.SGD(model_n331.parameters(), lr=0.1)
graph = load_graph(330)

loss_list, nodes, coordinates = train(nodes_num, model_n331, optimizer, epochs=500)
X_all = [coordinates[i, j, 0].item() for i in range(10) for j in range(52)]
Y_all = [coordinates[i, j, 1].item() for i in range(10) for j in range(52)]

v_connected, X, Y, v_coords = connected_with_v(20, nodes, coordinates)

'''print(connected_with_v(1, nodes, coordinates)[0])
print(sorted(graph[1]))
# sprawdzam poprawnosc funkcji connected_with_v'''


plt.scatter(X_all, Y_all, s=0.7)
plt.scatter(X, Y, c='green')
plt.scatter(v_coords[0], v_coords[1], c='red')
plt.legend(["niepolaczone z czerwonym", "polaczone z czerwonym"])
plt.show()
