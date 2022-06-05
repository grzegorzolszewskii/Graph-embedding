from bfs import bfs
from greedy_routing import greedy_routing
from graph_import import load_graph
import pandas as pd
from greedy_routing import eukl_dist
from greedy_routing import hyp_dist


def gr_success_rate(graph, coordinates, dist):
    success = 0
    for i in range(len(graph)):
        for j in range(len(graph)):
            if i != j:
                if bfs(graph, i, j) == greedy_routing(graph, coordinates, i, j, dist):
                    success += 1
                else:
                    print(bfs(graph, i, j), greedy_routing(graph, coordinates, i, j, dist))

    return success/(len(graph)**2 - len(graph))


if __name__ == '__main__':
    nodes_num = 46
    graph = load_graph(nodes_num, 'tree_graph')
    coordinates_eukl = pd.read_csv('eucl_2d', header=None, skiprows=[nodes_num+1])
    print(gr_success_rate(graph, coordinates_eukl, eukl_dist))
    print("--------------------------------------------------")

    coordinates_hyp = pd.read_csv('hyp_3d', header=None, skiprows=[nodes_num+1])
    print(gr_success_rate(graph, coordinates_hyp, hyp_dist))
    print("?")
