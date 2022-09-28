from graph_import import load_graph
from manifolds import Manifold
from embed import embed
import argparse


def find_best_emb(graph, manifold, dims, lrs, epochs, alpha, loops, max_loss):
    min_loss, coords, params = 100, None, None
    for loop in range(loops):
        for d in dims:
            for l in lrs:
                for e in epochs:
                    for a in alpha:
                        loss, weights = embed(graph, manifold, d, l, e, a, max_loss)
                        if loss < min_loss:
                            min_loss = loss
                            coords = weights
                            params = (manifold.manifold_type, d, l, e, a)

    return min_loss, coords, params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Szukanie najlepszego zanurzenia dla roznych parametrow")
    parser.add_argument('-g', '--graph', type=str, required=True, help='zanurzany graf')
    parser.add_argument('-gs', '--graph_size', type=int, required=True, help='rozmiar grafu')
    parser.add_argument('-m', '--manifold', type=str, required=True, help='przestrzen metryczna')
    parser.add_argument('-loss', '--max_loss', type=float, help='maksymalna wartosc funkcji kosztu')
    args = parser.parse_args()

    dims = [4, 5, 6]
    lrs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.25]
    epochs = [300]
    loops = 3
    alphas = [1, 3, 5, 7, 10, 12, 15]

    graph = load_graph(args.graph_size, args.graph)
    manifold = Manifold(args.manifold)
    max_loss = args.max_loss

    loss, coordinates, params = find_best_emb(graph, manifold, dims, lrs, epochs, alphas, loops, max_loss)
    dim = params[1]
    print(loss, params)

    with open('best_embedding', 'w') as file:
        for i in range(len(graph)):
            for j in range(dim):
                file.write(str(coordinates[i][j].item()))
                if j != dim-1:
                    file.write(', ')
            file.write('\n')
        file.write('\n')
        line = args.graph, params, loss
        file.write(str(line))
