# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 17:19
# @Author  : Chen Jinghan
# @File    : GradientDescent.py
import matplotlib.pyplot as plt
import numpy as np

from LiHongYi_ML.HW1 import DataPreprocessing


class PredictByGradientDescent:

    def __init__(self):
        pass

    def gradient_descent(self, x_lst, y_lst):
        """
        只考虑PM2.5预测PM2.5
        用Gradient Descent
        :return: bias 和 w
        """
        iteration = 1000
        lr = 0.00000001
        b = 1.0
        w_arr = np.ones(9 * 1)
        his_b = []
        his_w = []

        for i in range(iteration):
            b_grad = 0.0
            w_grad = np.zeros(9 * 1)
            # y=b+w1*x1+w2*x2+...+w9*x9
            for j in range(0, len(x_lst)):
                temp_feature = x_lst[j].iloc[9].values
                b_grad = b_grad + 2 * (y_lst[j] - (b + np.dot(temp_feature, w_arr))) * (-1)
                for k in range(len(w_grad)):
                    w_grad[k] = w_grad[k] + 2 * (y_lst[j] - (b + np.dot(temp_feature, w_arr))) * (-temp_feature[k])

            b = b - lr * b_grad
            his_b.append(b)
            for h in range(len(w_arr)):
                w_arr[h] = w_arr[h] - lr * w_grad[h]
            his_w.append(w_arr)

        print(b)
        print(w_arr)
        return b, w_arr, his_b, his_w





