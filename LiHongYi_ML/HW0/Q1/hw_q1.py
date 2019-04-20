# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 19:27
# @Author  : Chen Jinghan
# @File    : hw_q1.py
import numpy as np

"""
Q1 矩阵运算
读取matrixA.txt and matrixB.txt
进行矩阵乘法运算
由大到小排序后输出
"""


def get_data1():
    with open("data1","r") as f:
        data1 = f.read()
        # split by ','
        lst1 = data1.split(',')
        # str to int
        nums1 = [int(x) for x in lst1]
        # list to matrix
        mat1 = np.mat(nums1)
    return mat1


def get_data2():
    with open("data2","r") as f:
        start = 1
        for line in f.readlines():
            line = line.replace("\n","")
            lst2 = line.split(',')
            nums2 = [int(x) for x in lst2]
            data2 = np.matrix(nums2)
            if start == 1:
                matrix2 = data2
                start = 0
            else:
                matrix2 = np.vstack((matrix2, data2))
    return matrix2


def get_result():
    matrix3 = get_data1() * get_data2()
    res = matrix3.tolist()[0]
    print(sorted(res))
    return sorted(res)


get_result()