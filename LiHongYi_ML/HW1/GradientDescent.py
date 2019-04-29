# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 17:19
# @Author  : Chen Jinghan
# @File    : GradientDescent.py

import numpy as np


class PredictByGradientDescent:

    def __init__(self):
        pass

    def gradient_descent(self, x_lst, y_lst):
        """
        Gradient Descent
        not work don't know why.........
        :return: bias and w
        """
        iteration = 1000
        lr = 0.1
        b = 10
        w_arr = np.ones(len(x_lst[0])) * 10
        his_b = []
        his_w = []

        for i in range(iteration):
            b_grad = 0.0
            w_grad = np.zeros(len(x_lst[0]))
            # y=b+w1*x1+w2*x2+...+w9*x9
            for j in range(len(x_lst)):
                temp_feature = list(x_lst[j].values)
                print(temp_feature)
                b_grad = b_grad + 2 * (y_lst[j] - (b + np.dot(temp_feature, w_arr))) * (-1)
                # value of w
                for k in range(len(w_grad)):
                    print(i,j,k)
                    w_grad[k] = w_grad[k] + 2 * (y_lst[j] - (b + np.dot(temp_feature, w_arr))) * (-temp_feature[k])

            b = b - lr * b_grad
            his_b.append(b)
            for h in range(len(w_arr)):
                w_arr[h] = w_arr[h] - lr * w_grad[h]
            his_w.append(w_arr)

        print("b:  ", b)
        print("w_arr:  ", w_arr)
        print("his_b", b)
        return b, w_arr, his_b, his_w





