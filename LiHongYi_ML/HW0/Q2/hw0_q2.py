# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 20:48
# @Author  : Chen Jinghan
# @File    : hw0_q2.py
"""
Q2 图像处理
读取两个图片
使用后者异于前者的部分产生相同格式的新图片
pillow
"""
from PIL import Image


img1 = Image.open("lena.png")
img2 = Image.open("lena_modified.png")

x, y = img2.size
# print(img1.getpixel((0, 0)))
for i in range(0, x):
    for j in range(0, y):
        # 返回给定位置的像素值
        if img1.getpixel((i, j)) == img2.getpixel((i, j)):
            # putpixel将坐标为(i, j)的像素点变为(255,0,0)颜色，即红色
            img2.putpixel((i, j), 255)

# 神奇！
img2.save('res.png')