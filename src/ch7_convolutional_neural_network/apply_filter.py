# coding: utf-8
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet
from matplotlib.image import imread
from common.layers import Convolution





def filter_show(filters, nx=4, show_num=16):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """

    FN, C, FH, FW = filters.shape   # # 解包滤波器的形状：FN=滤波器数量, C=通道数, FH=滤波器高度, FW=滤波器宽度
    ny = int(np.ceil(show_num / nx)) # 含义：计算需要的行数，确保所有滤波器都能被显示

    fig = plt.figure()  # 创建一个新的图形窗口
    # # 调整子图之间的间距：left/right/bottom/top设置边界，hspace/wspace设置子图间距
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
   
    # 循环遍历要显示的滤波器数量
    for i in range(show_num):
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])   # 创建一个4x4网格中的第i+1个子图，xticks=[], yticks=[]表示不显示坐标轴刻度# 创建一个4x4网格中的第i+1个子图，xticks=[], yticks=[]表示不显示坐标轴刻度

        # # 显示第i个滤波器的第一个通道，使用灰度反转色图，最近邻插值
        ax.imshow(filters[i, 0], cmap = plt.cm.gray_r, interpolation="nearest")



network = SimpleConvNet(
    input_dim=(1,28,28), 
    conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
    hidden_size=100, output_size = 10, weight_init_std = 0.01
)

# 调用filter_show函数，显示网络第一层卷积层的权重W1的前16个滤波器
# 这里显示的是随机初始化后的滤波器（尚未训练）
filter_show(network.params['W1'], 16)   # # network.params['W1']是第一层卷积层的权重，形状为(30, 1, 5, 5)


network.load_params("params.pkl")   # 从文件中加载预训练好的网络参数（权重和偏置）

img = imread('../dataset/lena_gray.png') # 读取灰度图像（lena是经典的测试图像）
img = img.reshape(1, 1, *img.shape)  # 将图像reshape为卷积网络需要的4D格式， 转换为 (1, 1, 28, 28)

fig = plt.figure()  # 创建一个新的图形窗口，用于显示卷积后的输出

w_idx = 1

for i in range(16):
    w = network.params['W1'][i]
    b = 0  # network.params['b1'][i]

    # 将权重reshape为4D格式：(滤波器数量, 通道数, 高, 宽)
    w = w.reshape(1, *w.shape)  # 转换为 (1, 1, 5, 5)
    #b = b.reshape(1, *b.shape)
    conv_layer = Convolution(w, b)
    out = conv_layer.forward(img) # # 对输入图像进行前向传播（卷积运算），输出形状为(1, 1, H', W')， shape (1, 1, 28, 28)

    out = out.reshape(out.shape[2], out.shape[3])   # 将4D输出压缩为2D：(高, 宽)，便于显示

    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks= [])
    ax.imshow(out, cmap = plt.cm.gray_r, interpolation="nearest")



fig.show()



