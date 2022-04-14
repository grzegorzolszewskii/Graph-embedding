import pandas as pd
import random


def load_graph2(vertices_num, data):
    edges_set = {i: set() for i in range(vertices_num)}
    edges_pd = pd.read_csv(data, header=None, delimiter=" ")

    for i in range(len(edges_pd)):
        if edges_pd[0][i] < vertices_num and edges_pd[1][i] < vertices_num:
            edges_set[edges_pd[0][i]].add(edges_pd[1][i])
            edges_set[edges_pd[1][i]].add(edges_pd[0][i])
    print(edges_set)

    # edges_set[2].add(3)
    # sprawdzam poprawnosc - uzywajac powyzszej linijki tworze blad w danych
    for num1 in edges_set:
        for num2 in edges_set:
            if num2 in edges_set[num1] and num1 not in edges_set[num2]:
                raise ValueError("Blad w danych")

    # usuwam zbiory puste - po zmianie pliku zakladam ze ich nie ma
    for j in list(edges_set):
        if edges_set[j] == set():
            print("Ucinam niepolaczony wierzch nr: ", j)
            edges_set.pop(j)

    return edges_set


if __name__ == '__main__':
    print(load_graph2(20, data='tree_graph'))

