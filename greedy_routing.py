from graph_import_start0 import load_graph2
from model import Model
from manifolds import Manifold
import torch as th
from train_function import train
from draw_by_weights import draw2
import pandas as pd


def dist(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


# szukam drogi z a do b
def greedy_routing(graph, coordinates, a, b):
    a_coords = (coordinates[0][a], coordinates[1][a])
    b_coords = (coordinates[0][b], coordinates[1][b])

    v = a
    prev = None
    path = [(v, dist(a_coords, b_coords))]

    for k in range(10):
        v_connected_coords = {i: (0, 0, 0) for i in graph[v]}  # 2 wspolrzedne oraz odleglosc od b
        v_further_neighbours = dict()
        for w in graph[v]:
            for i in graph[w]:
                v_further_neighbours[i] = (0, 0, 0)  # 2wsp, odl od b oraz wierzch laczacy z v

        for w in v_connected_coords:  # ustalamy koordynaty polaczonych z v
            tmp_coords = coordinates[0][w], coordinates[1][w]
            if w == prev:   # jezeli w jest poprzednim - unikamy robienia kolek - ustawiamy sztucznie duzy dystans
                v_connected_coords[w] = (tmp_coords[0], tmp_coords[1], 100)
            v_connected_coords[w] = (tmp_coords[0], tmp_coords[1], dist(tmp_coords, b_coords))
            if w == b:  # jezeli w jest wierzch docelowym - konczymy dzialanie programu, zwracamy sciezke
                path.append((w, 0))
                return [path[i][0] for i in range(len(path))]

        for w in v_further_neighbours:  # ustalamy koordynaty polaczonych z polaczonymi z v
            tmp_coords = coordinates[0][w], coordinates[1][w]
            if w == prev:   # jezeli w jest poprzednim - unikamy robienia kolek - ustawiamy sztucznie duzy dystans
                v_further_neighbours[w] = (tmp_coords[0], tmp_coords[1], 100)
            v_further_neighbours[w] = (tmp_coords[0], tmp_coords[1], dist(tmp_coords, b_coords))
            if w == b:  # jezeli w jest wierzch docelowym - konczymy dzialanie programu, zwracamy sciezke
                path.append((v, dist(tmp_coords, b_coords)))
                path.append((w, 0))
                return [path[i][0] for i in range(len(path))]

        min_dist = 100
        for w in v_connected_coords:  # szuakmy ktory w polaczony z v ma najmniejsza odleglosc do docelowego
            if min_dist > v_connected_coords[w][2] > 0:
                min_dist = v_connected_coords[w][2]
                min_v = w

        for w in v_further_neighbours:
            if min_dist > v_further_neighbours[w][2] > 0:
                min_dist = v_further_neighbours[w][2]
                min_v = w

        if min_v in graph[v]:
            path.append((min_v, min_dist))
        else:
            path.append((v, dist((coordinates[1][v], coordinates[0][v]), b_coords)))
            path.append((min_v, min_dist))


        # przejscie - zmieniam v na kolejny wierzcholek
        print(v_connected_coords)
        print(v_further_neighbours)
        print("ide z", v, "do ", min_v)
        prev = v
        v = min_v


if __name__ == "__main__":
    nodes_num = 46
    graph = load_graph2(nodes_num, data='tree_graph')
    coordinates = pd.read_csv('good_embedding', header=None)

    print(greedy_routing(graph, coordinates, 20, 40))
    # draw2(graph, 0, coordinates)
