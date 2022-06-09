from graph_import import load_graph
from manifolds import Manifold
from embed import embed


def find_best_emb(graph, manifold, dims, lrs, epochs, alpha, loops, max_loss):
    min_loss, coords, params = 100, None, None
    for loop in range(loops):
        for d in dims:
            for l in lrs:
                for e in epochs:
                    for a in alpha:
                        loss, weights = embed(graph, manifold, d, l, e, a, max_loss)

                        print(loss, l, a)
                        if loss < min_loss:
                            min_loss = loss
                            coords = weights
                            params = (manifold.manifold_type, d, l, e, a)

    return min_loss, coords, params


if __name__ == '__main__':
    nodes_num = 46
    graph = load_graph(nodes_num, data='tree_graph')

    manifold_euclidean = Manifold('euclidean')
    manifold_lorentz = Manifold('lorentz')

    dims_euclidean = [6]
    dims_lorentz = [4, 5, 6]

    lrs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.25]
    # lrs = [0.07, 0.08, 0.09]
    epochs = [300]
    loops = 3
    alphas = [1, 3, 5, 7, 10, 12, 15]
    # alphas = [6, 7, 8]

    loss, coordinates, params = find_best_emb(graph, manifold_euclidean, dims_euclidean,
                                              lrs, epochs, alphas, loops, max_loss=3.74)
    dim = params[1]
    print(loss, params)

    if loss < 3.74:
        with open('eucl_6d', 'w') as file:
            for i in range(len(graph)):
                for j in range(dim):
                    file.write(str(coordinates[i][j].item()))
                    if j != dim-1:
                        file.write(', ')
                file.write('\n')
            file.write('\n')
            line = params, loss
            file.write(str(line))
