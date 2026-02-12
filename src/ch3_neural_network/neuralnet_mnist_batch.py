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
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y



def predict_with_relu(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']


    a1 = np.dot(x, w1) + b1
    z1 = relu(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = relu(a2)

    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y





x, t = get_data()
network = init_network()

batch_size = 100 # 批量大小
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict_with_relu(network, x_batch)  # y_batch.shape = (100, 10)
    p = np.argmax(y_batch, axis=1)  # p.shape = (100,)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))