import matplotlib.pyplot as plt
from model import Model
from manifolds import Manifold
import torch as th
from train_function import train
from graph_import_start0 import load_graph2


def draw2(graph, v, weights):
    v_coords = (weights[v][0].item(), weights[v][1].item())
    X_connected = []
    Y_connected = []
    for w in graph[v]:
        X_connected.append(weights[w][0].item())
        Y_connected.append(weights[w][1].item())

    X_all = [weights[i][0].item() for i in range(0, len(graph))]
    Y_all = [weights[i][1].item() for i in range(0, len(graph))]

    for i in range(len(graph)):
        for j in range(len(graph)):
            if i in graph[j] or j in graph[i]:
                X_c = (weights[i][0].item(), weights[j][0].item())
                Y_c = (weights[i][1].item(), weights[j][1].item())
                plt.plot(X_c, Y_c, c='black', linewidth=0.3)

    plt.scatter(X_all, Y_all, c='blue')
    plt.scatter(X_connected, Y_connected, c='green')
    plt.scatter(v_coords[0], v_coords[1], c='red')
    plt.legend(["niepolaczone z czerwonym", "polaczone z czerwonym"])

    plt.show()


if __name__ == '__main__':
    graph = load_graph2(20, data='tree_graph')
    nodes_num = len(graph)

    eucl = Manifold('euclidean')
    model = Model(eucl, nodes_num, 2)
    optimizer = th.optim.SGD(model.parameters(), lr=0.1)

    loss_list, weights = train(graph, model, optimizer, epochs=400, max_loss=3.75)
    draw2(graph, 0, weights)
