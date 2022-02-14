import torch as th
e = th.zeros(size=(10, 52))
for i in range(0, 10):
    for j in range(0, 52):
        if j == 0:
            e[i, j] = 99 + i
        else:
            e[i, j] = i + j

print("Wektor e: ", e)
o = e.narrow(1, 1, e.size(1) - 1)  # wierzcholki bez 1 kolumny
s = e.narrow(1, 0, 1).expand_as(o)

print("Wektor o: ", o)
print("Wektor s: ", s)
print(len(s[0]))


def distance(u, v):
    return ((u - v).pow(2)).pow(1 / 2).sum(dim=-1)


dist = distance(o, s)
print(dist.squeeze(-1))
