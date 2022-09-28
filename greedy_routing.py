from graph_import import load_graph
import pandas as pd
from acosh import acosh
from math import acosh


def eukl_dist(p1, p2):
    if len(p1) != len(p2):
        raise ValueError("Different dimensions")

    d = 0
    for i in range(len(p1)):
        d += (p1[i] - p2[i])**2
    return d


def hyp_dist(p1, p2):
    if len(p1) != len(p2):
        raise ValueError("Different dimensions")

    product = - p1[0] * p2[0]
    for i in range(1, len(p1)):
        product += p1[i] * p2[i]
    return acosh(-product)


def greedy_routing(graph, coordinates, a, b, dist):
    if a == b:
        return [a]

    dim = coordinates.shape[1]
    a_coords = [float(coordinates[i][a]) for i in range(dim)]
    b_coords = [float(coordinates[i][b]) for i in range(dim)]

    prev = None
    v = a
    path = [(v, dist(a_coords, b_coords))]
    min_v = None

    while True:
        v_connected_coords = {i: [0 for j in range(dim+1)] for i in graph[v]}

        for w in v_connected_coords:  # saving coordinates of the vertices that are connected to v
            tmp_coords = [float(coordinates[i][w]) for i in range(dim)]

            if w == b:
                path.append((w, 0))
                return [path[i][0] for i in range(len(path))]

            for i in range(dim):
                v_connected_coords[w][i] = float(coordinates[i][w])
                v_connected_coords[w][dim] = dist(tmp_coords, b_coords)

        min_dist = 10000
        if prev == v:   # if it glitches - return path
            return [path[i][0] for i in range(len(path)-1)]

        if prev is not None:
            v_connected_coords[prev][dim] = min_dist  # big distance for prev - algorithm does not turn back

        for w in v_connected_coords:
            if min_dist > v_connected_coords[w][dim] > 0:
                min_dist = v_connected_coords[w][dim]
                min_v = w

        path.append((min_v, min_dist))
        prev = v
        v = min_v
