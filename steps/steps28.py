if '__file__' in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import numpy as np
import dezero as dz

def rosenbrock(x0, x1):
    y = 100*(x1 - x0**2)**2 + (1 - x0)**2
    return y

x0 = dz.Variable(np.array(0.0))
x1 = dz.Variable(np.array(2.0))

y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)

# 梯度下降法的实现 最小值为(1,1)
# x0 = dz.Variable(np.array(0.0))
# x1 = dz.Variable(np.array(2.0))
# lr = 0.001
# iters = 10000

# for i in range(iters):
#     print(x0, x1)
#     y = rosenbrock(x0, x1)
#     x0.cleargrad()
#     x1.cleargrad()
#     y.backward()

#     x0.data -= lr * x0.grad
#     x1.data -= lr * x1.grad

