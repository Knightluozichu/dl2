import numpy as np


if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
from dezero import Config

def f(x:Variable)->Variable:
    y = x ** 4 - 2 * x ** 2
    return y

# x = Variable(np.array(2.0))
# y = f(x)
# x.cleargrad()
# y.backward(create_graph=True)
# print(x.grad)

# gx = x.grad
# x.cleargrad()
# gx.backward()
# print(x.grad)
print("-----")
print(Config.enable_backprop)
x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i,x)
    
    y = f(x)
    x.cleargrad()
    # print(y.creator)
    y.backward(create_graph=True)
    
    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad
    x.data = x.data - gx.data / gx2.data