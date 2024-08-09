if "__file__" in globals():
    import os,sys
    sys.path.append(os.path.join(os.path.dirname(__file__),".."))

import numpy as np
import dezero.layers as L
import dezero.functions as F
from dezero import Layer

# model = Layer()
# model.l1 = L.Linear(5)
# model.l2 = L.Linear(3)

# def predice(model, x):
#     y = model.l1(x)
#     y = F.sigmoid(y)
#     y = model.l2(y)
#     return y

# for p in model.params():
#     print(p)

# model.cleargrads()

# class TwoLayerNet(Layer):
#     def __init__(self, hidden_size, out_size):
#         super().__init__()
#         self.l1 = L.Linear(hidden_size)
#         self.l2 = L.Linear(out_size)
    
#     def forward(self, x):
#         y = F.sigmoid(self.l1(x))
#         y = self.l2(y)
#         return y
    

from dezero import Model,Variable
from dezero.models import MLP

class TwoLayerNet(Model):
    def __init__(self,hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = self.l1(x)
        y = F.sigmoid(y)
        y = self.l2(y)
        return y
    
x_np = np.random.rand(5 ,10)
x = Variable(x_np, name='x')
model = TwoLayerNet(100,10)
model.plot(x)

# model = TwoLayerNet(100,10)
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

lr = 0.2
max_iter = 10000
hidden_size = 10  
model = TwoLayerNet(hidden_size,1)

# for i in range(max_iter):
#     y_pred = model(x)
#     loss = F.mean_squared_error(y, y_pred)
#     model.cleargrads()
#     loss.backward()
    
#     for p in model.params():
#         p.data -= lr * p.grad.data
    
#     if i  % 1000 == 0:
#         print(loss)


m = MLP((8,16,1))
m.plot(x)

for i in range(max_iter):
    y_pred = m(x)
    loss = F.mean_squared_error(y, y_pred)
    m.cleargrads()
    loss.backward()

    for p in m.params():
        p.data -= lr * p.grad.data
    
    if i % 1000 == 0:
        print(loss)


