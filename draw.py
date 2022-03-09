import matplotlib.pyplot as plt
from model import Model
from manifolds import Euclidean
import torch as th
from train_function import train

ver_num = 330
eucl = Euclidean()
model_n331 = Model(eucl, 331, 2)
optimizer = th.optim.SGD(model_n331.parameters(), lr=0.1)
ret = train(ver_num, model_n331, optimizer, epochs=100)

wierzcholki = ret[1]
wspolrzedne = ret[2]
print(wspolrzedne)

X = [wspolrzedne[i, j, 0].item() for i in range(10) for j in range(50)]
Y = [wspolrzedne[i, j, 1].item() for i in range(10) for j in range(50)]
print(X)
print(Y)

print(wierzcholki)

plt.scatter(X, Y, s=0.5)
plt.show()
# zrobic tak - wybrac jeden wierzcholek - on i polaczone z nim zrobic innym kolorem np. zielonym
