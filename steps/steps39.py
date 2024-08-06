if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x0 = Variable(np.array([[1.0,2.0,3.0],[3.0,4.0,5.0]]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)
y.backward()
print(x1.grad)