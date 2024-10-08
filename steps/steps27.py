if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero as dz
from utils import plot_dot_graph

class Sin(dz.Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self,gy):
        x= self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
    
def sin(x):
    return Sin()(x)

# x = dz.Variable(np.array(np.pi/4))
# y = sin(x)
# y.backward()

# print(y.data)
# print(x.grad)

# 泰勒展开sin
def my_sin(x, threshold=1e-4):
    y = 0
    for i in range(100000):
        c = (-1)**i / np.math.factorial(2*i+1)
        t = c * x**(2*i+1)
        y = y + t
        if np.abs(t.data) < threshold:
            print(f'{i+1} times')
            break
    return y

x = dz.Variable(np.array(np.pi/4))
y = my_sin(x)
y.backward()

print(y.data)
print(x.grad)
x.name='x'
y.name='y'

plot_dot_graph(y, verbose=False, to_file='goldstein.png')