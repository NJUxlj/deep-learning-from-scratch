# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    """
    数值微分（用“中心差分”近似导数）。

    目标：近似计算函数 f 在点 x 处的导数 f'(x)。
    中心差分公式：
        f'(x) ≈ ( f(x + h) - f(x - h) ) / (2h)
    相比“前向差分” (f(x+h)-f(x))/h，中心差分的误差通常更小、更稳定。

    参数：
    - f: 一元函数（可接收标量/NumPy数组；但这里的 x 会传入标量）
    - x: 求导点（标量）
    """
    h = 1e-4  # 微小步长（0.0001）；h 太大→近似粗糙，h 太小→浮点舍入误差显著
    return (f(x + h) - f(x - h)) / (2 * h)  # 返回在 x 处的导数近似值


def function_1(x):
    """
    被研究的目标函数 f(x)。
    这里是一个二次函数：f(x) = 0.01 x^2 + 0.1 x
    """
    return 0.01 * x**2 + 0.1 * x


def tangent_line(f, x):
    """
    生成函数 f 在点 x 处的切线函数（返回一个可调用对象）。

    切线的一般形式可以写成：
        y = d * t + b
    其中：
    - d 是切线斜率（也就是 f 在 x 处的导数 f'(x)）
    - b 是截距

    因为切线必须经过点 (x, f(x))，代入 t = x 得：
        f(x) = d * x + b
    所以截距：
        b = f(x) - d * x
    """
    d = numerical_diff(f, x)  # d ≈ f'(x)：用数值微分得到切线斜率（导数）
    print(d)  # 打印斜率（便于观察在该点的导数值）
    b = f(x) - d * x  # b：切线截距，确保切线经过 (x, f(x))
    return lambda t: d * t + b  # 返回切线函数；t 可以是标量或 NumPy 数组
     
# 生成 x 轴上的一系列采样点：[0, 20)，步长 0.1
# np.arange(start, stop, step) 产生等间距数列（stop 不包含在内）
x = np.arange(0.0, 20.0, 0.1)

# 计算每个采样点对应的函数值 f(x)，得到曲线上的 y 坐标（NumPy 向量化计算）
y = function_1(x)

# 设置坐标轴标签（方便读图）
plt.xlabel("x")
plt.ylabel("f(x)")

# 在 x = 5 处求函数的切线，并得到“切线函数” tf(t)
# tf 是一个函数：传入 t（可为数组）会返回切线上的 y 值
tf = tangent_line(function_1, 5)

# 把整段 x 采样点带入切线函数，得到切线对应的 y 值序列
y2 = tf(x)

# 绘制原函数曲线 y = f(x)
plt.plot(x, y)

# 绘制切线曲线 y2（在 x=5 处与原函数相切）
plt.plot(x, y2)

# 弹出图形窗口显示结果
plt.show()