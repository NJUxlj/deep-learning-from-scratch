# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient_2d(f, X):
    """
    计算二维输入 X 的数值梯度的“兼容封装”。

    这个函数主要处理两种常见输入形态：
    - **X 是一维向量**（X.ndim == 1）：直接对该向量求梯度，返回形状与 X 相同的一维数组
    - **X 是二维矩阵**（X.ndim != 1，通常是 (N, D)）：
      把 X 看作 N 个样本（或 N 组参数向量），对每一行分别求梯度，
      最终得到与 X 同形状的梯度矩阵 grad

    说明：这里复用 `_numerical_gradient_1d` 来计算“单个向量”的梯度，
    然后在二维情况下逐行调用它。
    """
    if X.ndim == 1:
        # 输入本身就是一维向量：直接返回对该向量的梯度
        return _numerical_gradient_1d(f, X)
    else:
        # 输入是二维（或更高维的分支在此处不处理）：为每一行（每个样本）创建对应的梯度
        grad = np.zeros_like(X)
        
        # enumerate(X) 会按“第一维”迭代：
        # - idx：行号
        # - x：X[idx]，也就是一维向量（这一行）
        for idx, x in enumerate(X):
            # 对每一行单独计算梯度，并写回 grad 的对应行
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    """
    计算函数 f 在点 x 处的数值梯度（支持任意维度的 NumPy 数组 x）。

    这里的“梯度”指对 x 的每个元素分别求偏导，最后得到与 x 形状相同的数组 grad：
        grad[idx] ≈ ∂f/∂x[idx]

    做法：对每个元素 x[idx] 做一次“中心差分”扰动：
        ∂f/∂x[idx] ≈ ( f(x[idx]+h) - f(x[idx]-h) ) / (2h)

    注意：该实现会在循环中“原地修改 x”的某个元素来计算 f(x±h)，算完后必须还原，
    否则后续元素的梯度会在被污染的 x 上计算，结果会错误。
    """
    h = 1e-4  # 微小步长（0.0001）
    grad = np.zeros_like(x)  # 创建与 x 同形状的数组，用于保存每个位置的偏导数

    # np.nditer 用于遍历任意维数组的每个元素（包含多维索引信息）
    # - flags=['multi_index']：让迭代器提供当前元素的多维索引 it.multi_index（如 (i,j,k)）
    # - op_flags=['readwrite']：允许对被遍历的数组 x 做原地读写（本函数需要临时改 x[idx]）
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    # 逐元素遍历 x，对每个元素分别计算中心差分近似的偏导
    while not it.finished:
        idx = it.multi_index        # 当前元素的索引（元组），适用于任意维度
        tmp_val = x[idx]            # 先保存原始值，方便之后还原

        # 计算 f(x+h)：只对当前 idx 位置加上 h，其余位置保持不变
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)                 # f(x+h)

        # 计算 f(x-h)：只对当前 idx 位置减去 h
        x[idx] = tmp_val - h
        fxh2 = f(x)                 # f(x-h)

        # 中心差分得到该位置的偏导数近似值
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 还原 x[idx]，避免影响下一次迭代（否则后续偏导会在错误的 x 上计算）
        x[idx] = tmp_val

        # 迭代器移动到下一个元素
        it.iternext()

    return grad  # 返回与 x 同形状的梯度数组