# coding: utf-8
import numpy as np

class SGD:

    """随机梯度下降法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr

    
    def update(self, params ,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]





class Momentum:
    """动量法（Momentum）
    
    想象一个小球从山坡上滚下来：

    普通梯度下降 ｜ Momentum 优化器 
    小球每一步只看当前坡度 | 小球有惯性，会记住之前的运动方向 
    遇到小坑洼可能卡住   |  惯性帮助小球冲过小坑洼 
    方向变化剧烈，路径曲折  | 路径更平滑，收敛更快

    实际效果 ：

        - ✅ 加速收敛 ：如果梯度方向一致，速度会不断叠加，跑得越来越快
        - ✅ 抑制震荡 ：如果梯度方向来回震荡，正负抵消，速度趋于稳定
        - ✅ 逃离局部最优 ：惯性帮助"冲出"一些浅的局部最小值
    
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum   # 动量系数，通常=0.9。 他决定"惯性"有多大； 0.9 意味着：之前的运动方向保留 90% 的权重， 当前梯度只占 10% 的影响
        self.v = None   # v （速度） ：相当于小球的"记忆"，记录了之前的运动方向

    
    def update(self, params, grads):
        if self.v is None:
            self.v= {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key]  - self.lr * grads[key]   # v = momentum × v_旧 - lr × 梯度

            params[key] += self.v[key]





class Nesterov:

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]




class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None


    def update(self):
        pass











class AdaGrad:
    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None


        def update(self):
            pass

    




class Adam:

    """Adam"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None


        def update(self):
            pass
