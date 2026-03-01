# coding: utf-8
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)



# 减少学习数据
x_train = x_train[:1000]
t_train = t_train[:1000]


max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
