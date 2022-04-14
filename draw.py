import matplotlib.pyplot as plt
from model import Model
from manifolds import Manifold
import torch as th
from train_function import train
from graph_import_start1 import load_graph
from graph_import_start0 import load_graph2


def draw(graph, v, nodes, coordinates):  # argumentami sa macierze: 10x52, 10x52x2
    v_connected = []
    nodes_set = {nodes[i, j].item() for i in range(10) for j in range(52)}

    if v not in nodes_set:
        raise ValueError("Wierzcholek nie znalazl sie w ostatnim batchu")

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

    X_all = [coordinates[i, j, 0].item() for i in range(10) for j in range(52)]
    Y_all = [coordinates[i, j, 1].item() for i in range(10) for j in range(52)]

    # print(v_connected)
    # print(sorted(graph[v]))
    # print(loss_list)

    plt.scatter(X_all, Y_all, s=0.7)
    plt.scatter(X, Y, c='green')
    plt.scatter(v_coords[0], v_coords[1], c='red')
    plt.legend(["niepolaczone z czerwonym", "polaczone z czerwonym"])
    plt.show()


if __name__ == '__main__':
    graph = load_graph2(20, data='tree_graph')
    nodes_num = len(graph)

    eucl = Manifold('euclidean')
    model = Model(eucl, nodes_num, 2)
    optimizer = th.optim.SGD(model.parameters(), lr=0.1)

    loss_list, nodes, coordinates, weights = train(graph, model, optimizer, epochs=300)
    draw(graph, 0, nodes, coordinates)

    # zmniejszenie learning rate'a nic nie daje - loss function przestaje sie zmieniac, min to okolo 65 (lr=0.1)
    # KLUCZOWE zwiekszenie lr do 0.5 daje najlepsze wartosci loss_fun bliskie 50 (300 epok wystarczy)

