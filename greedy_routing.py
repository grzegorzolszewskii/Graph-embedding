from graph_import import load_graph
from model import Model
from manifolds import Manifold
import torch as th
from train_function import train


def dist(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


# szukam drogi z a do b
def greedy_routing(nodes, coordinates, a, b, nodes_num=330):
    graph = load_graph(nodes_num)
    v = a
    path = [v]

    b_coords = 0  # szukamy wspolrzedne b
    for i in range(10):
        for j in range(52):
            if nodes[i, j].item() == b:
                b_coords = (coordinates[i, j, 0].item(), coordinates[i, j, 1].item())
    if b_coords == 0:
        raise AttributeError("Nie znaleziono wierzcholka, do ktorego droge chcemy znalezc")

    for i in range(10):  # tu docelowo while(True)
        v_connected_coords = {i: (0, 0, 0) for i in graph[v]}  # 2 wspolrzedne oraz odleglosc od b
        for w in v_connected_coords:
            for i in range(10):
                for j in range(52):
                    if nodes[i, j].item() == w:
                        tmp_coords = coordinates[i, j, 0].item(), coordinates[i, j, 1].item()
                        v_connected_coords[w] = (tmp_coords[0], tmp_coords[1], dist(tmp_coords, b_coords))
            if w == b:
                path.append((w, 0))
                return path

        min_dist = 100
        for w in v_connected_coords:  # szuakmy min w[2]
            if min_dist > v_connected_coords[w][2] > 0:
                min_dist = v_connected_coords[w][2]
                min_v = w

        path.append((min_v, min_dist))
        print("ide do wierzcholka: ", min_v, "zdystansowanego od docelowego o: ", min_dist)
        print(path)
        v = min_v


if __name__ == "__main__":
    nodes_num = 330
    eucl = Manifold('euclidean')
    model = Model(eucl, nodes_num + 1, 2)
    optimizer = th.optim.SGD(model.parameters(), lr=0.5)
    graph = load_graph(nodes_num)
    loss_list, nodes, coordinates = train(nodes_num, model, optimizer, epochs=300)
    print(greedy_routing(nodes, coordinates, 150, 230))
