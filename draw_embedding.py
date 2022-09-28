import matplotlib.pyplot as plt
import torch as th
import pandas as pd


def to_poincare_ball(coordinates):
    torch_tensor = th.tensor(coordinates.values)    # pd.df to th.tensor
    x = torch_tensor.clone()
    d = x.size(-1) - 1
    narrowed = x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)
    return pd.DataFrame(narrowed.numpy())           # th.tensor to pd.df


def draw(graph, coordinates, V=None):
    if V:
        VX_coords = [coordinates[0][i] for i in V]
        VY_coords = [coordinates[1][i] for i in V]

    X_all = [coordinates[0][i] for i in range(0, len(graph))]
    Y_all = [coordinates[1][i] for i in range(0, len(graph))]

    for i in range(len(graph)):
        for j in range(len(graph)):
            if i in graph[j] or j in graph[i]:
                X_c = (coordinates[0][i].item(), coordinates[0][j].item())
                Y_c = (coordinates[1][i].item(), coordinates[1][j].item())
                plt.plot(X_c, Y_c, c='black', linewidth=0.3)

    plt.scatter(X_all, Y_all, c='blue')
    if V:
        plt.scatter(VX_coords, VY_coords, c='red')

    plt.show()
