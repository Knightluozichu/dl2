if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
from utils import plot_dot_graph

x = Variable(np.array(1.0))
y  = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 2 
for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

# plot_dot_graph(y, verbose=False, to_file='tanh_forward.png')

gx = x.grad
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose=True, to_file='tanh_iters_7.png')

# %%
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import numpy as np
# import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
from utils import plot_dot_graph

# x=2.0, y=x^2,z=y'^3+y
x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()
z = gx ** 3 + y
z.backward()
print(x.grad)

# %%
