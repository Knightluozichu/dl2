if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

# 切片
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.get_item(x,1)
print(y)
print(x.shape)
y.backward()
print(x.grad)

indices = np.array([0,0,1])
y = F.get_item(x, indices)
print(y)

y = x[1]
print(y)

y = x[:,2]
print(y)

# softmax
from dezero.models import MLP

model = MLP((10,3))

x = np.array([[0.2,-0.4]])
y = model(x)
print(y)

from dezero import Variable,as_variable
import dezero.functions as F

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

x = Variable(np.array([[0.2,-0.4]]))
y = model(x)
p = softmax1d(y)
print(y)
print(p)

def softmax_simple(x, axis = 1):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y, axis=axis, keepdims=True)
    return y / sum_y

# 交叉熵误差
def softmax_vross_entropy_simple(x,t):
    x, t = as_variable(x), as_variable(t)    
    N = x.shape[0]

    p  = softmax_simple(x)
    p = F.clip(p, 1e-15,1.0)
    log_p = F.log(p)
    tlog_p = log_p[np.arange(N),t.data]
    y = -1 * sum(tlog_p) / N
    return y


x = Variable(np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]]))
t = np.array([2,0,1,0])
y = model(x)
loss = softmax_vross_entropy_simple(y,t)
print(loss)
