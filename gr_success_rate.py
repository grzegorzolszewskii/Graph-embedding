from bfs import bfs
from greedy_routing import greedy_routing
from graph_import import load_graph
import pandas as pd
from greedy_routing import eucl_dist
from greedy_routing import hyp_dist
import random

random.seed(1)


def gr_success_rate(graph, coordinates, dist, pairs):
    success = 0
    for pair in pairs:
        bfs_path = bfs(graph, pair[0], pair[1])
        gr_path = greedy_routing(graph, coordinates, pair[0], pair[1], dist)

        if gr_path:
            if bfs_path == gr_path or len(bfs_path) == len(gr_path):    # moga byc rozne najkrotsze sciezki
                success += 1

    return success / len(pairs)


if __name__ == '__main__':
    nodes_num = 1180
    graph = load_graph(nodes_num, 'mammals_graph.csv', delimiter=',')
    coordinates = pd.read_csv('eucl_50d', header=None, skiprows=[nodes_num + 1])

    pairs_num = 1000
    pairs = [(0, 0) for i in range(pairs_num)]
    for i in range(pairs_num):
        while pairs[i][0] == pairs[i][1]:
            pairs[i] = (random.randrange(0, nodes_num), random.randrange(0, nodes_num))
    print(pairs)

    print(gr_success_rate(graph, coordinates, eucl_dist, pairs))
