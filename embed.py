import torch as th
from graph_import import load_graph
from manifolds import Manifold
from model import Model
from train_function import train
from rsgd import RiemannianSGD


# dla roznych parametrow otrzymuje loss fun i wspolrzedne - wykonuje jedno zanurzenie
def embed(graph, manifold, dim, lr, epoch, alpha, max_loss):
    model = Model(manifold, len(graph), dim, alpha)
    if manifold.manifold_type == 'euclidean':
        optimizer = th.optim.SGD(model.parameters(), lr=lr)
    if manifold.manifold_type == 'lorentz':
        optimizer = RiemannianSGD(model.optim_params(), lr=lr)

    loss, weights = train(graph, model, optimizer, epochs=epoch, max_loss=max_loss)
    return loss, weights


if __name__ == '__main__':
    graph = load_graph(46, 'tree_graph')
    manifold = Manifold('euclidean')
    dim = 2
    lr = 1
    epoch = 300
    alpha = 1
    max_loss = 4.9

    loss, coordinates = embed(graph, manifold, dim, lr, epoch, alpha, max_loss)
    print(loss)

    if loss < max_loss:
        with open('single_embed', 'w') as file:
            for i in range(len(graph)):
                for j in range(dim):
                    file.write(str(coordinates[i][j].item()))
                    if j != dim-1:
                        file.write(', ')
                file.write('\n')
            file.write('\n')
            file.write(str((manifold.manifold_type, dim, lr, epoch, alpha, loss)))
