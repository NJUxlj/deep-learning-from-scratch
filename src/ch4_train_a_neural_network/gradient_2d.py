# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
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




def numerical_gradient_advanced(f, x):
    h = 1e-5
    grad = np.zeros_like(x)

    print("x.size: ", x.size)

    for idx in range(x.size):
        tmp_val = x[idx]
        
        x_plus_h = tmp_val + h
        x_minus_h = tmp_val - h

        fxh1 = f(x_plus_h)
        fxh2 = f(x_minus_h)

        grad[idx] = (fxh1 - fxh2) / (2*h)

    return grad


        



def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


def function_2(x):
    '''
    类似于 f(x0, x1) = x0^2 + x1^2 的函数
    '''
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x  # 由于 f(x) = d*x + b，所以 b = f(x) - d*x

    # 这里为什么要把 y 算出来： 因为我们现在想拿到切线 l(x) , 而不是 f(x), 而 l(t) = d*t + b
    return lambda t: d*t + y
     
if __name__ == '__main__':
    # 构造二维平面上的采样网格：
    # x0, x1 分别代表二维自变量的两个坐标轴取值范围
    x0 = np.arange(-2, 2.5, 0.25)   # length = (2.5-(-2))/0.25  = 18
    x1 = np.arange(-2, 2.5, 0.25)

    # meshgrid 会把一维坐标轴“扩展成网格坐标矩阵”
    # - X 的形状与 Y 相同，都是 (len(x1), len(x0)) = (18, 18)
    # - 网格上的每个点 (X[i,j], Y[i,j]) 就是一组二维输入 (x0, x1)
    X, Y = np.meshgrid(x0, x1)
    
    # 为了方便把“网格矩阵”喂给梯度函数，这里把 X, Y 展平成一维：
    # - flatten 后 X、Y 都变成长度为 N 的向量（N = 网格点数量） = 18 * 18 = 324
    X = X.flatten()
    Y = Y.flatten()
    print(X.shape, Y.shape)

    # 把每个网格点的 (X[k], Y[k]) 组合成二维向量：
    # np.array([X, Y]) 的形状是 (2, N)，转置后变成 (N, 2)
    # 也就是 N 个二维点：[[x0_0, x1_0],
    #                  [x0_1, x1_1],
    #                  ...
    #                  [x0_{N-1}, x1_{N-1}]]
    # 形状为 (324, 2)
    grad = numerical_gradient(function_2, np.array([X, Y]).T)
    
    # 画“梯度场/向量场”：
    # 对标量函数 f(x0, x1)，梯度 ∇f 指向“函数值上升最快”的方向
    # 负梯度 -∇f 指向“函数值下降最快”的方向（梯度下降法就是沿这个方向走）
    #
    # 注意形状对应关系：
    # - 这里 grad 的形状会是 (N, 2)，即每个点一个二维梯度向量 [df/dx0, df/dx1]
    # - 因此每个点的 x0 分量应该用 grad[:, 0]，x1 分量用 grad[:, 1]
    # - 下面这行若写成 -grad[0], -grad[1]，语义上是“取第 0/1 个点的梯度”，形状也不匹配
    #   正确的分量选取通常应为：-grad[:, 0], -grad[:, 1]
    plt.figure()
    # U、V 必须与 X、Y 等长（都是 N 个点的分量），所以这里取 grad 的两列
    plt.quiver(X, Y, -grad[:, 0], -grad[:, 1], angles="xy", color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()

    # 只有在 plot/quiver 等对象设置了 label 时，legend 才会显示内容；
    # 否则这里通常会得到一个空图例或警告。
    plt.legend()
    plt.draw()
    plt.show()





'''
`meshgrid` 做的事可以理解成：**把两条“坐标轴刻度”（`x0`、`x1`）扩展成一张“坐标表格”**，从而列出平面上所有网格点的 \((x0, x1)\) 组合。

## `x0`、`x1` 是什么？
- **`x0`**：x 轴（第 1 个自变量）的取值列表
- **`x1`**：y 轴（第 2 个自变量）的取值列表

在你的代码里：

- `x0 = np.arange(-2, 2.5, 0.25)`
- `x1 = np.arange(-2, 2.5, 0.25)`

它们都是一维数组，表示“网格在两个方向分别取哪些坐标”。

## `meshgrid(x0, x1)` 返回的 `X, Y` 是什么？
`X` 和 `Y` 都是二维数组（矩阵），形状一样，都是：

- 行数 = `len(x1)`
- 列数 = `len(x0)`

并且满足：

- **`X[i, j] = x0[j]`**（每一行都把 `x0` 原样复制一遍）
- **`Y[i, j] = x1[i]`**（每一列都把 `x1` 原样复制一遍）

换句话说：
- `X` 提供每个网格点的 **x 坐标**
- `Y` 提供每个网格点的 **y 坐标**
- 网格上的点就是：\((X[i,j],\ Y[i,j])\)

## 一个超小例子（看得最清楚）
假设：

```python
x0 = np.array([10, 20, 30])
x1 = np.array([1, 2])
X, Y = np.meshgrid(x0, x1)
```

那么：

- `X` 会是：

```python
[[10, 20, 30],
 [10, 20, 30]]
```

- `Y` 会是：

```python
[[1, 1, 1],
 [2, 2, 2]]
```

对应的网格点（把同位置配对）就是：

- (10, 1), (20, 1), (30, 1)
- (10, 2), (20, 2), (30, 2)

也就是把 `x0` 和 `x1` 做了“笛卡尔积”列举。

## 它们和你后面 `flatten()` 的关系
你代码里：

- `X = X.flatten()`
- `Y = Y.flatten()`

会把上面那种二维“坐标表格”压平成一维列表，变成“点集形式”：

例子里会得到：

- `X.flatten()` = `[10, 20, 30, 10, 20, 30]`
- `Y.flatten()` = `[ 1,  1,  1,  2,  2,  2]`

再配对后依然表示同一批网格点，只是从矩阵形式换成了列表形式，方便一次性喂给 `quiver` 或组装成 \((N,2)\) 的输入。

如果你愿意，我也可以直接在你当前的 `gradient_2d.py` 里加一个“小尺寸 meshgrid 示例打印”（用很小的 `x0/x1`），这样你运行脚本时就能在终端看到 `X/Y` 的具体样子。


'''