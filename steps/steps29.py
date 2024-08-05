if '__file__' in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import numpy as np
import dezero as dz

# 牛顿法
def f(x0):
    y = x0**4 - 2*x0**2
    return y

def f2(x):
    y  = 12*x**2 - 4
    return y

x0 = dz.Variable(np.array(2.0))

iters = 10

for i in range(iters):
    print(i, x0)

    y = f(x0)
    x0.cleargrad()
    y.backward()

    x0.data -= x0.grad / f2(x0).data

