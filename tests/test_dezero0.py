import unittest
import numpy as np

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
from utils import numerical_diff

# x = Variable(np.array(1.0))
# y = (x + 3) ** 2
# y.backward()

# print(y)
# print(x.grad)

def sphere(x,y):
    z = x ** 2 + y ** 2
    return z

def matyas(x,y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

def square(x):
    return x ** 2

class TestDeZero0(unittest.TestCase):
    def test_sphere_forward(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(2.0))
        z = sphere(x,y)
        assert z.data == 5.0
    
    def test_sphere_backward(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(2.0))
        z = sphere(x,y)
        z.backward()
        assert x.grad == 2.0
        assert y.grad == 4.0

    def test_gradient_check(self):
        x = Variable(np.random.randn(1))
        z = square(x)
        z.backward()
        num_grad = numerical_diff(square,x)
        flg =  np.allclose(x.grad,num_grad)
        self.assertTrue(flg)
        # assert numerical_diff(square,x,1e-4) == x.grad



unittest.main()