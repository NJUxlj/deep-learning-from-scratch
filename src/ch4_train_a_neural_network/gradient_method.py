# coding: utf-8
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x   # 初始点, 比如 [-3.0, 4.0]
    x_history = []  # 记录每次迭代的点

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)  # shape = (2,)
        x -= lr * grad  # shape = (2,)

    return x, np.array(x_history)


def function_2(x):
    '''
    f(x0, x1) = x0^2 + x1^2
    '''
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

# 画出 x 轴（y=0，蓝色虚线）
plt.plot([-5, 5], [0, 0], '--b')
# 画出 y 轴（x=0，蓝色虚线）
plt.plot([0, 0], [-5, 5], '--b')
# 画出梯度下降过程中每一步的轨迹（每步的 (x0, x1) 位置用圆点表示）
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

# 设置 x 轴显示范围
plt.xlim(-3.5, 3.5)
# 设置 y 轴显示范围
plt.ylim(-4.5, 4.5)
# x 轴标签
plt.xlabel("X0")
# y 轴标签
plt.ylabel("X1")
# 显示图像
plt.show()