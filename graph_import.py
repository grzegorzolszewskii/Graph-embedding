import pandas as pd
import random


def load_graph(nodes_num, data):
    edges_set = {i: set() for i in range(nodes_num)}
    edges_pd = pd.read_csv(data, header=None, delimiter=" ")

    for i in range(len(edges_pd)):
        if edges_pd[0][i] < nodes_num and edges_pd[1][i] < nodes_num:
            edges_set[edges_pd[0][i]].add(edges_pd[1][i])
            edges_set[edges_pd[1][i]].add(edges_pd[0][i])

    for num1 in edges_set:
        for num2 in edges_set:
            if num2 in edges_set[num1] and num1 not in edges_set[num2]:
                raise ValueError("Blad w danych")

    # usuwam zbiory puste
    for j in list(edges_set):
        if edges_set[j] == set():
            print("Ucinam niepolaczony wierzch nr: ", j)
            edges_set.pop(j)

    return edges_set
