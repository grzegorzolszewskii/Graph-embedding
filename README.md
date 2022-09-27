# Graph embedding
Zanurzanie grafów w przestrzeń euklidesową oraz hiperboliczną jest jednym z rozwiązań problemu najkrótszej ścieżki. Zanurzanie przypisuje grafom dodatkową własność - każdy wierzchołek po zanurzeniu grafu posiada współrzędne w pewnej przestrzeni. Dodanie tej własności pozwala szukać najkrótszej ścieżki między wierzchołkami za pomocą algorytmu greedy routing, który ma znacznie mniejszą złożoność czasową niż popularne algorytmy szukania najkrótszych ścieżek w grafie jak np. BFS lub algorytm Dijkstry.

Program wykonuje zanurzenie oraz korzystając z pewnych reguł matematycznych dopasowuje zanurzenie w taki sposób, aby algorytm greedy routing działał jak najlepiej na zanurzeniu danego grafu. Porównywane są także zanurzenia w przestrzeń euklidesową z zanurzeniami w przestrzeń hiperboliczną.

W projekcie wykorzystuję popularne algorytmy oraz metody stosowane w uczeniu maszynowym za pomocą biblioteki PyTorch.
