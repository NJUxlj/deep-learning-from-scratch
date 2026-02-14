# coding: utf-8
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 为了导入父目录的文件而进行的设定
sys.path.append(str(Path(__file__).parent.parent.parent))  # 为了导入父目录的文件而进行的设定
from layer_naive import *



apple = 100
apple_num = 2
tax = 1.1


mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()


apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)



print("backward")
dprice =1


dapple_price, dtax = mul_tax_layer.backward(dprice)


dapple, dapple_num = mul_apple_layer.backward(dapple_price)


print(f"price: {price}")
print(f"dapple: {dapple}, dapple_num: {dapple_num}, dtax: {dtax}")