from graph_import import load_graph2
from model import Model
from manifolds import Manifold
import torch as th
from train_function import train
from draw_by_weights import draw2
import pandas as pd
from bfs import bfs
from acosh import acosh
from manifolds import LorentzDot


def eukl_dist(p1, p2):
    if len(p1) != len(p2):
        raise ValueError("Wektory maja rozne wymiary!")
    d = 0
    for i in range(len(p1)):
        d += (p1[i] - p2[i])**2
    return d


def hyp_dist(p1, p2):
    if len(p1) != len(p2):
        raise ValueError("Wektory maja rozne wymiary!")
    d = -LorentzDot.apply(p1, p2)
    d.data.clamp_(min=1)
    return acosh(d, 1e-5)


# szukam drogi z a do b
def greedy_routing(graph, coordinates, a, b, dist):
    if a == b:
        return [a]

    dim = coordinates.shape[1]
    a_coords = [float(coordinates[i][a]) for i in range(dim)]
    b_coords = [float(coordinates[i][b]) for i in range(dim)]

    prev = None
    v = a
    path = [(v, dist(a_coords, b_coords))]

    for k in range(10):  # powinno byc while True
        v_connected_coords = {i: [0 for j in range(dim+1)] for i in graph[v]}  # 2 wspolrzedne oraz odleglosc od b

        for w in v_connected_coords:  # ustalamy koordynaty polaczonych z v
            tmp_coords = [float(coordinates[i][w]) for i in range(dim)]

            for i in range(dim):
                v_connected_coords[w][i] = float(coordinates[i][w])
            v_connected_coords[w][dim] = dist(tmp_coords, b_coords)

            if w == b:
                path.append((w, 0))
                return [path[i][0] for i in range(len(path))]

        min_dist = 100
        if prev == v:   # jezeli sie blokuje to zwracam path
            return [path[i][0] for i in range(len(path)-1)]

        if prev is not None:
            v_connected_coords[prev][dim] = min_dist  # ustalamy poprzednikowi duzy dystans, aby gr nie zawracal

        for w in v_connected_coords:  # szuakmy ktory w polaczony z v ma najmniejsza odleglosc do docelowego
            if min_dist > v_connected_coords[w][dim] > 0:
                min_dist = v_connected_coords[w][dim]
                min_v = w

        path.append((min_v, min_dist))

        # przejscie - zmieniam v na kolejny wierzcholek
        # print(v_connected_coords)
        # print("ide z", v, "do ", min_v)
        prev = v
        v = min_v


if __name__ == "__main__":
    nodes_num = 46
    graph = load_graph2(nodes_num, data='tree_graph')

    coordinates_eukl = pd.read_csv('best_embedding', header=None, skiprows=[nodes_num+1])
    coordinates_hyp = pd.read_csv('hyperbolic_embedding', header=None, skiprows=[nodes_num+1])

    print(bfs(graph, 43, 11))
    print(greedy_routing(graph, coordinates_eukl, 43, 11, eukl_dist))

    # draw2(graph, 0, coordinates)
