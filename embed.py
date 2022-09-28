import torch as th
from graph_import import load_graph
from manifolds import Manifold
from model import Model
from train_function import train
from rsgd import RiemannianSGD
from draw_embedding import draw, to_poincare_ball
import argparse
import pandas as pd


# dla roznych parametrow otrzymuje wartosc loss i wspolrzedne - wykonuje jedno zanurzenie
def embed(graph, manifold, dim, lr, epoch, alpha, max_loss):
    model = Model(manifold, len(graph), dim, alpha)
    if manifold.manifold_type == 'euclidean':
        optimizer = th.optim.SGD(model.parameters(), lr=lr)
    if manifold.manifold_type == 'lorentz':
        optimizer = RiemannianSGD(model.optim_params(), lr=lr)

    loss, weights = train(graph, model, optimizer, epochs=epoch, max_loss=max_loss)
    return loss, weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pojedyncze zanurzenie wybranego grafu w przestrzen metryczna")
    parser.add_argument('-g', '--graph', type=str, required=True, help='zanurzany graf')
    parser.add_argument('-gs', '--graph_size', type=int, required=True, help='rozmiar grafu')
    parser.add_argument('-m', '--manifold', type=str, required=True, help='przestrzen metryczna')
    parser.add_argument('-dim', '--dim', type=int, required=True, help='liczba wymiarow przestrzeni')
    parser.add_argument('-lr', '--lr', type=float, required=True, help='wspolczynnik uczenia')
    parser.add_argument('-e', '--epochs', type=int, help='liczba epok')
    parser.add_argument('-a', '--alpha', type=float, help='parametr alpha')
    parser.add_argument('-loss', '--max_loss', type=float, help='maksymalna wartosc funkcji kosztu')
    args = parser.parse_args()

    graph = load_graph(args.graph_size, args.graph)
    manifold = Manifold(args.manifold)
    loss, coordinates = embed(graph, manifold, args.dim, args.lr, args.epochs, args.alpha, args.max_loss)

    with open('single_embed', 'w') as file:
        for i in range(args.graph_size):
            for j in range(args.dim):
                file.write(str(coordinates[i][j].item()))
                if j != args.dim-1:
                    file.write(', ')
            file.write('\n')
        file.write('\n')
        file.write(str((args.graph, args.manifold, args.dim, args.lr, args.epochs, args.alpha, loss)))

    coordinates = pd.read_csv('single_embed', header=None, skiprows=[args.graph_size+1])
    if args.dim == 2 and args.manifold == 'euclidean':
        draw(graph, coordinates)
    if args.dim == 3 and args.manifold == 'lorentz':
        draw(graph, to_poincare_ball(coordinates))

