# coding: utf-8
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 为了导入父目录的文件而进行的设定
sys.path.append(str(Path(__file__).parent.parent.parent))  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    
    def predict(self, x):
        '''
        x: (N, D)
        W1: (D, H)
        b1: (H,)
        W2: (H, O)
        b2: (O,)

        return: (N, O)

        '''
        W1, W2 = self.params["W1"], self.params["W2"]

        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1  # shape = (N, H)
        z1 = sigmoid(a1)  # shape = (N, H)

        a2 = np.dot(z1, W2) + b2  # shape = (N, O)

        y = softmax(a2)  # shape = (N, O)

        return y


    def loss(self, x,  t):
        '''
        x: 输入数据 (N, D)
        t: 标签 (N, O)

        return: 损失值
        '''
        y = self.predict(x)  # shape = (N, O)
        loss = cross_entropy_error(y, t)
        return loss



    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads


    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']


        grads = {}

        batch_num = x.shape[0]

        # forward 
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)


        # backward
        



