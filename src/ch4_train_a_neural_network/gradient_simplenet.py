# coding: utf-8
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 为了导入父目录中的文件而进行的设定
sys.path.append(str(Path(__file__).parent.parent.parent))  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

# f 就是 net.loss, 再对 W 求梯度时， x 和 t 这样的输入可以视为常数值。
# 为什么 f 可以对 W 求梯度？明明看起来这两者没关系啊？ 以为 net.loss 里面调用了 self.predict, 里面又调用了 W, 所以 W 是 f 中的一个的变量。

'''
也就是说 loss 的值是由 self.W 决定的。而 numerical_gradient 正是在每次调用 f(...) 之前，把 self.W（同一个数组对象）悄悄改了。
所以真实依赖链是：
f(w)（参数没用）
依赖 net.loss(x,t)
依赖 net.predict(x)
依赖 net.W（被数值梯度函数原地扰动）

'''
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)