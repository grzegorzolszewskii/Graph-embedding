import torch as th
from graph_import import load_graph
from manifolds import Manifold
from model import Model
from train_function import train
from rsgd import RiemannianSGD
from draw_embedding import draw, to_poincare_ball
import argparse
import pandas as pd


# single embedding, plot if possible
def embed(graph, manifold, dim, lr, epoch, alpha, max_loss):
    model = Model(manifold, len(graph), dim, alpha)
    if manifold.manifold_type == 'euclidean':
        optimizer = th.optim.SGD(model.parameters(), lr=lr)
    if manifold.manifold_type == 'lorentz':
        optimizer = RiemannianSGD(model.optim_params(), lr=lr)

    loss, weights = train(graph, model, optimizer, epochs=epoch, max_loss=max_loss)
    return loss, weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single embedding")
    parser.add_argument('-g', '--graph', type=str, required=True, help='embedded graph')
    parser.add_argument('-gs', '--graph_size', type=int, required=True, help='graph size')
    parser.add_argument('-m', '--manifold', type=str, help='metric space', default='euclidean')
    parser.add_argument('-dim', '--dim', type=int, help='dimensions', default=2)
    parser.add_argument('-lr', '--lr', type=float, help='learing rate', default=0.5)
    parser.add_argument('-e', '--epochs', type=int, help='epochs', default=300)
    parser.add_argument('-a', '--alpha', type=float, help='alpha', default=1)
    parser.add_argument('-loss', '--max_loss', type=float, help='maximum loss fun value after embedding', default=3)
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
