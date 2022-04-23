import matplotlib.pyplot as plt
from model import Model
from manifolds import Manifold
import torch as th
from train_function import train
from graph_import_start0 import load_graph2
import pandas as pd


def draw2(graph, v, coordinates):
    v_coords = (coordinates[0][v], coordinates[1][v])
    X_connected = []
    Y_connected = []
    for w in graph[v]:
        X_connected.append(coordinates[0][w])
        Y_connected.append(coordinates[1][w])

    X_all = [coordinates[0][i] for i in range(0, len(graph))]
    Y_all = [coordinates[1][i] for i in range(0, len(graph))]

    for i in range(len(graph)):
        for j in range(len(graph)):
            if i in graph[j] or j in graph[i]:
                X_c = (coordinates[0][i].item(), coordinates[0][j].item())
                Y_c = (coordinates[1][i].item(), coordinates[1][j].item())
                plt.plot(X_c, Y_c, c='black', linewidth=0.3)

    plt.scatter(X_all, Y_all, c='blue')
    plt.scatter(X_connected, Y_connected, c='green')
    plt.scatter(v_coords[0], v_coords[1], c='red')
    plt.legend(["niepolaczone z czerwonym", "polaczone z czerwonym"])

    plt.show()


if __name__ == '__main__':
    nodes_num = 46
    graph = load_graph2(nodes_num, data='tree_graph')
    coordinates = pd.read_csv('good_embedding_dim2', header=None)
    # W PD.DF INDEKSOWANIE JEST ODWROTNE !!!

    draw2(graph, 43, coordinates)
