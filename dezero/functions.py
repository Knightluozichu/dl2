import numpy as np
from dezero.core import Function


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gys):
        x, = self.inputs
        gs = gys * cos(x)
        return gs
    

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gys):
        x, = self.inputs
        gs = gys * -sin(x)
        return gs
    

def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self,x):
        y = np.tanh(x)
        return y
    
    def backward(self, gys):
        y = self.outputs[0]()
        gy =  gys * (1 - y * y)
        return gy
    
def tanh(x):
    return Tanh()(x)