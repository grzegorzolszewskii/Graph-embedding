from bfs import bfs
from greedy_routing import greedy_routing
from graph_import import load_graph
import pandas as pd
import random
from greedy_routing import eukl_dist, hyp_dist
import argparse

random.seed(1)

parser = argparse.ArgumentParser(description="Greedy routing success rate")
parser.add_argument('-g', '--graph', type=str, required=True, help='graph')
parser.add_argument('-gs', '--graph_size', type=int, required=True, help='graph size')
parser.add_argument('-eg', '--embedded_graph', type=str, required=True, help='coordinates of the embedded graph '
                                                                             'vertices')
parser.add_argument('-m', '--manifold', type=str, help='metric space')
parser.add_argument('-p', '--pairs_num', type=int, help='how many vertices pairs for checking greedy routing'
                                                        'success rate')
args = parser.parse_args()


# are the greedy routing shortest paths the same as the BFS shortest paths?

def gr_success_rate(graph, coordinates, dist, pairs_num):
    nodes_num = len(graph)
    pairs = [(0, 0) for i in range(pairs_num)]
    for i in range(pairs_num):
        while pairs[i][0] == pairs[i][1]:
            pairs[i] = (random.randrange(0, nodes_num), random.randrange(0, nodes_num))

    success = 0
    for pair in pairs:
        bfs_path = bfs(graph, pair[0], pair[1])
        gr_path = greedy_routing(graph, coordinates, pair[0], pair[1], dist)

        if gr_path:
            if bfs_path == gr_path or len(bfs_path) == len(gr_path):    # moga byc rozne najkrotsze sciezki
                success += 1

    return success / len(pairs)


if __name__ == '__main__':
    coordinates = pd.read_csv(args.embedded_graph, header=None, skiprows=[args.graph_size + 1])
    graph = load_graph(args.graph_size, args.graph)
    if args.manifold == 'euclidean':
        print("Greedy routing success rate: ", gr_success_rate(graph, coordinates, eukl_dist, args.pairs_num))
    if args.manifold == 'lorentz':
        print("Greedy routing success rate: ", gr_success_rate(graph, coordinates, hyp_dist, args.pairs_num))
