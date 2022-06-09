from bfs import bfs
from greedy_routing import greedy_routing
from graph_import import load_graph
import pandas as pd
from greedy_routing import eukl_dist
from greedy_routing import hyp_dist
import random


random.seed(1)


def gr_success_rate(graph, coordinates, dist, pairs):
    success = 0
    for pair in pairs:
        if bfs(graph, pair[0], pair[1]) == greedy_routing(graph, coordinates, pair[0], pair[1], dist):
            success += 1
        else:
            print(bfs(graph, pair[0], pair[1]), greedy_routing(graph, coordinates, pair[0], pair[1], dist))
    return success/len(pairs)


if __name__ == '__main__':
    nodes_num = 46
    graph = load_graph(nodes_num, 'tree_graph')
    coordinates_eukl = pd.read_csv('eucl_6d', header=None, skiprows=[nodes_num + 1])

    pairs_num = 1000
    pairs = [(0, 0) for i in range(pairs_num)]
    for i in range(pairs_num):
        while pairs[i][0] == pairs[i][1]:
            pairs[i] = (random.randrange(0, nodes_num), random.randrange(0, nodes_num))
    print(pairs)

    print(gr_success_rate(graph, coordinates_eukl, eukl_dist, pairs))





