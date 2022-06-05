import torch as th
import random as rand
from graph_import import load_graph
from manifolds import Manifold
from model import Model
from rsgd import RiemannianSGD


def train(graph, model, optimizer, epochs=50, max_loss=3.5):
    nodes_num = len(graph)
    loss_list = [0 for i in range(epochs)]

    for epoch in range(epochs):
        for batch in range(0, int(len(graph) / 10)):

            inputs = th.randint(nodes_num, (10, 52))    # th.randint bierze z [0, a)
            for i in range(0, 10):                      # random.randint bierze z [a, b]
                node = 10 * batch + i
                node_connected = rand.choice(tuple(graph[node]))
                inputs[i, 0] = node
                inputs[i, 1] = node_connected
                for j in range(2, 52):      # jest 50 losowan - duza szansa ze wylosuje sie polaczony
                    inputs[i, j] = rand.randint(0, nodes_num-1)
                    while inputs[i, j].item() in graph[node]:
                        inputs[i, j] = rand.randint(0, nodes_num-1)

            optimizer.zero_grad()       # za kazdym razem chcemy nowy gradient
            preds = model(inputs)
            target = th.zeros(10).long()

            loss = model.loss(preds, target=target, size_average=True)
            loss_list[epoch] += loss.item()

            loss.backward()  # obliczenie pochodnych
            optimizer.step()  # dodaj pochodne do pozycji wierzch

        if epoch > epochs/2:
            if loss_list[epoch] < min(loss_list[:epoch]):
                if loss_list[epoch] <= max_loss:
                    print("Koniec zanurzania dla loss rownego: ", loss_list[epoch])
                    return loss_list, model.model.weight

    return loss_list, model.model.weight


if __name__ == '__main__':
    # pojedyncze zanurzenie
    nodes_num = 46
    graph = load_graph(nodes_num, data='tree_graph')

    dim = 2
    lr = 0.4
    epochs = 300
    eucl = Manifold('euclidean')
    model = Model(eucl, nodes_num, dim)
    optimizer = th.optim.SGD(model.parameters(), lr=lr)  # dla prz euklidesowej
    # optimizer = RiemannianSGD(model.optim_params(), lr=lr)  # dla lorentza

    loss_list, weights = train(graph, model, optimizer, epochs=epochs, max_loss=5)
    loss_list_changed = [item for item in loss_list if item > 0]
    print(min(loss_list_changed))
    print(loss_list)
    print(weights)
