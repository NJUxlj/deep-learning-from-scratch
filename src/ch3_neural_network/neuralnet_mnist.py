# coding: utf-8
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import pickle
from src.dataset.mnist import load_mnist
from src.common.functions import sigmoid, softmax, relu


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    weight_path = str(Path(__file__).parent) + "/sample_weight.pkl"

    with open(weight_path, 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()   # x.shape = (10000, 784), t.shape = (10000,)
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])   # y.shape = (10,)
    p= np.argmax(y) # 获取概率最高的类别索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))