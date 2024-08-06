if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1.0,2.0,3.0],[3.0,4.0,5.0]]))
y = x.reshape((6,))
z = x.transpose()
# y.backward(retain_grad=True)
# print(y.shape,y.grad.shape)
# print(x.grad.shape,x.shape)
print(x)
print(y)
print(z)

# Transpose

x = Variable(np.random.rand(2,3,4))
print(x.shape)
y = x.transpose(axes=(2,0,1))
print(y.shape)
y.backward()
