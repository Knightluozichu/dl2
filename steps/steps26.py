#%%

import numpy as np
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dezero as dz
from utils import get_dot_graph, plot_dot_graph

# x0 = dz.Variable(np.array(1.0))
# x1 = dz.Variable(np.array(1.0))
# y = x0 + x1

# x0.name = 'x0'
# x1.name = 'x1'
# y.name = 'y'

# txt = get_dot_graph(y, verbose=True)
# print(txt)
# file = os.path.join(os.path.dirname(__file__), '../graphviz/sample.dot')
# with open(file, 'w') as o:
#     o.write(txt)

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

x = dz.Variable(np.array(1.0))
y = dz.Variable(np.array(1.0))
z = goldstein(x, y)  # sphere(x, y) / matyas(x, y)
z.backward()

x.name='x'
y.name='y'
z.name='z'
plot_dot_graph(z, verbose=False, to_file='goldstein.png')

# %%
