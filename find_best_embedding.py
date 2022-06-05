import torch as th
from graph_import import load_graph
from manifolds import Manifold
from model import Model
from train_function import train
from rsgd import RiemannianSGD


def find_best_emb(graph, manifold, dims, lrs, epochs, loops, alpha):
    loss_with_params = dict()
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

                        loss_list, weights = train(graph, model, optimizer, epochs=e, max_loss=3.5)
                        loss_with_params[(min(filter(lambda x: x > 0, loss_list)))] = (manifold, d, l, e, a, weights)

    return min(loss_with_params), loss_with_params[min(loss_with_params)]


if __name__ == '__main__':
    nodes_num = 46
    graph = load_graph(nodes_num, data='tree_graph')

    manifold_euclidean = Manifold('euclidean')
    manifold_lorentz = Manifold('lorentz')

    dims_euclidean = [2, 3, 4, 5]
    dims_lorentz = [3]

    lrs = [0.01, 0.03]
    epochs = [300]
    loops = 3
    alpha = [10, 12, 15, 17, 20]

    loss, params = find_best_emb(graph, manifold_lorentz, dims_lorentz, lrs, epochs, loops, alpha)
    dim = params[1]
    coordinates = params[5]

    print(loss)
    if loss < 5.26:
        with open('hyp_3d', 'w') as file:
            for i in range(len(graph)):
                for j in range(dim):
                    file.write(str(coordinates[i][j].item()))
                    if j != dim-1:
                        file.write(', ')
                file.write('\n')
            file.write('\n')
            file.write(str((params[0].manifold_type, params[1], params[2], params[3], params[4], loss)))
