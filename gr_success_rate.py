from bfs import bfs
from greedy_routing import greedy_routing
from graph_import import load_graph2
import pandas as pd


def gr_success_rate(graph, coordinates):
    success = 0
    for i in range(len(graph)):
        for j in range(len(graph)):
            if i != j:
                if bfs(graph, i, j) == greedy_routing(graph, coordinates, i, j):
                    success += 1
                else:
                    print(bfs(graph, i, j), greedy_routing(graph, coordinates, i, j))

    return success/(len(graph)**2 - len(graph))


if __name__ == '__main__':
    graph = load_graph2(46, 'tree_graph')
    coordinates = pd.read_csv('best_embedding', header=None)
    print(gr_success_rate(graph, coordinates))
