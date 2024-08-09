import dezero
import numpy as np
from dezero import Function
from dezero import as_variable, as_array
from dezero import Variable
from dezero import cuda
import utils


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

class Exp(Function):
    def forward(self,x):
        y = np.exp(x)
        return y
    
    def backward(self, gys):
        y = self.outputs[0]() #weakref
        gx = gys * y
        return gx
    
def exp(x):
    return Exp()(x)

class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y
    
    def backward(self, gys):
        x, = self.inputs
        gx = gys / x
        return gx
    
def log(x):
    return Log()(x)

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

# 张量操作：reshape/ transpose / get_item / expand_dims / flatten / sum_to / broadcast_to

class Reshape(Function):
    def __init__(self,shape:tuple):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gys):
        return reshape(gys, self.x_shape)

def reshape(x:Variable, shape:tuple):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def __init__(self,axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y
    
    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
    
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)
        
def transpose(x:Variable, axes=None):
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y
    
    def backward(self, gys):
        x , = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gys)
    
class GetItemGrad(Function):
    def __init__(self,slices,in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self,gy):
        xp = dezero.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)
        # print(xp)
        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)
    

def get_item(x, slices):
    return GetItem(slices)(x)


def expand_dims(x, axis):
    x = as_variable(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))

def flatten(x):
    """将输入展平，不影响批处理大小"""
    return reshape(x, (x.shape[0],-1))

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
    

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
        
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
    
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

def average(x, axis=None, keepdims=False):
    x = as_variable(x)
    y = sum(x, axis, keepdims)
    return y / np.array(x.size).astype(y.dtype)

class MatMul(Function):
    def forward(self, x, W):
        y = np.dot(x, W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T,gy)
        return gx, gW
    
def matmul(x, W):
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * ( 2. / len(diff))
        gx1 = -gx0
        return gx0, gx1
    
def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None
    return y

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y
    
    def backward(self, gy):
        x,W,b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb
    
def linear(x, W, b=None):
    return Linear()(x, W, b)

class Sigmoid(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)
        y = 1 / (1 + np.exp(-x))
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx
    
def sigmoid(x):
    return Sigmoid()(x)
