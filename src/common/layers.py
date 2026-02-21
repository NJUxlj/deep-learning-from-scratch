# coding: utf-8
import numpy as np
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 为了导入父目录的文件而进行的设定
sys.path.append(str(Path(__file__).parent.parent.parent))  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.util import im2col, col2im




class Relu:
    def __init__(self):
        self.mask = None



    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <=0)
        out = x.copy()
        out[self.mask]  = 0
        return out


    def backward(self, dout: np.ndarray) -> np.ndarray:
        '''
        从 Relu 的上游传过来一个  dL/dz, 记作 dout， 现在我们要计算 dL/dz * dz/dx = dL/dx
        '''
        dout[self.mask] = 0  # relu 的导数不是 1 就是 0， 并且在 0 点不可导
        dx  =dout

        return dx








class Sigmoid:
    def __init__(self):
        self.out = None
    

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out) # y = sigmoid(x), dy/dx = y * (1 - y)
        return dx




class Affine:
    def __init__(self, W, b):
        self.W = W  
        self.b = b

        self.x = None
        self.original_x_shape = None

        # 权重和偏置参数的导数
        self.dW = None
        self.db = None


    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)   # shape (N, D)
        self.x = x

        out = np.dot(self.x, self.W) + self.b   # shape (N, M)
        return out


    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape) # 还原输入数据的形状（对应张量）
        return dx







class SoftmaxWithLoss:

    def __init__(self):
        self.loss = None
        self.y = None # softmax 的输出
        self.t = None # 监督数据

    def forward(self, x, t):
        pass