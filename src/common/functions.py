# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad


def softmax(x):
    """
    Softmax：把输入向量/矩阵转换为概率分布（最后一维上各元素非负且和为 1）。

    - **输入**: x 可以是一维向量，或带 batch 的二维/多维数组
    - **处理维度**: 在最后一维（axis=-1）上做 softmax
    - **数值稳定性**: 先减去该维度的最大值，避免 exp 计算溢出（不改变 softmax 结果）
    """
    # 数值稳定性：对最后一维做平移（减去最大值），避免 np.exp(x) 出现上溢
    x = x - np.max(x, axis=-1, keepdims=True)
    # 按最后一维归一化：exp 后除以该维度上的和，使结果每行（或每个样本）加起来为 1
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    """
    交叉熵误差（Cross Entropy Error），常用于分类任务的损失函数。

    - **y**: 模型输出的概率分布（shape 通常为 (batch, num_classes)）
    - **t**: 监督标签；既可以是
      - **标签索引**（shape 为 (batch,) 或 (batch, 1)），例如 [2, 0, 3]
      - **one-hot 向量**（shape 为 (batch, num_classes)）
    - **返回**: 对 batch 取平均的交叉熵损失
    """
    if y.ndim == 1:
        # 如果是单样本（1 维），统一 reshape 成 (1, 类别数)，方便用同一套 batch 代码处理
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 如果教师数据是one-hot向量，则转换为正确标签的索引
    if t.size == y.size:
        # one-hot -> 标签索引：例如 [0,0,1,0] -> 2
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    
    # 取出每个样本在“正确类别”上的预测概率 y[i, t[i]]
    # 对其取 log 得到对数似然；加 1e-7 避免 log(0) 变成 -inf
    # 最后加负号并对 batch 求平均：L = - (1/N) * Σ log p(correct)
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)