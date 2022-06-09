import torch as th
from graph_import import load_graph
from manifolds import Manifold
from model import Model
from train_function import train
from rsgd import RiemannianSGD


def find_best_emb(graph, manifold, dims, lrs, epochs, alpha, loops, max_loss):
    min_loss, coords, params = 100, None, None
    for loop in range(loops):
        for d in dims:
            for l in lrs:
                for e in epochs:
                    for a in alpha:
                        model = Model(manifold, nodes_num, d, a)

                        if manifold.manifold_type == 'euclidean':
                            optimizer = th.optim.SGD(model.parameters(), lr=l)
                        if manifold.manifold_type == 'lorentz':
                            optimizer = RiemannianSGD(model.optim_params(), lr=l)

                        loss, weights = train(graph, model, optimizer, epochs=e,
                                              max_loss=max_loss)
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

    dims_euclidean = [2]
    dims_lorentz = [4, 5, 6]

    lrs = [0.01, 0.05, 0.09, 0.3, 0.7, 1, 1.2]
    epochs = [300]
    loops = 7
    alphas = [1, 5, 10, 15]

    loss, coordinates, params = find_best_emb(graph, manifold_euclidean, dims_euclidean,
                                              lrs, epochs, alphas, loops, max_loss=5)
    dim = params[1]
    print(loss, params)

    if loss < 5:
        with open('test_file', 'w') as file:
            for i in range(len(graph)):
                for j in range(dim):
                    file.write(str(coordinates[i][j].item()))
                    if j != dim-1:
                        file.write(', ')
                file.write('\n')
            file.write('\n')
            line = params, loss
            file.write(str(line))
