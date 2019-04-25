# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 21:51
# @Author  : Chen Jinghan
# @File    : SGD.py
import numpy as np
import matplotlib.pyplot as plt


class StochasricGradientDescent:
    def __init__(self):
        pass

    def SDG(self, x_lst, y_lst):
        lr = 0.0001
        iteration = 1000
        b = 1.0
        w_arr = np.ones(9 * 1)

        for i in range(iteration):
            b_grad = 0.0
            w_grad = np.zeros(9 * 1)

            for j in range(len(x_lst)):
                temp_feature = x_lst[j].iloc[9].values
                b_grad = 2 * (y_lst[j] - (b + np.dot(w_arr * temp_feature))) * (-1)
                b = b - lr * b_grad
                for k in range(len(w_grad)):
                    w_grad[k] = 2 * (y_lst[j] - (b + np.dot(w_arr * x_lst))) * (-temp_feature[k])
                    w_arr[k] = w_arr[k] - lr * w_grad[k]

        return b, w_arr