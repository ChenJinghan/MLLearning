# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 19:56
# @Author  : Chen Jinghan
# @File    : Adagrad.py

import numpy as np

class Adagrad:
    def __init__(self):
        pass

    def adagrad(self, x_lst, y_lst):
        # learning rate
        lr = 0.0001
        iteration = 250
        b = 0
        w_arr = np.zeros(9 * 1)
        b_sum = 0
        w_sum_arr = np.zeros(9 * 1)

        b_his = []
        w_his = []

        for i in range(iteration):
            b_grad = 0.0
            w_grad = np.zeros(9 * 1)
            # y=b+w1*x1+w2*x2+...+w9*x9
            # 用每个training data 进行训练
            for j in range(0, len(x_lst)):
                temp_feature = x_lst[i].iloc[9].values
                b_grad = b_grad + 2 * (y_lst[j] - (b + np.dot(temp_feature, w_arr))) * (-1)
                # 依次更新每个w
                for k in range(len(w_grad)):
                    w_grad[k] = w_grad[k] + 2 * (y_lst[j] - (b + np.dot(temp_feature, w_arr))) * (-temp_feature[k])

            b_sum += b_grad ** 2
            w_sum_arr += w_grad ** 2

            b = b - lr * b_grad/np.sqrt(b_sum)
            b_his.append(b)
            # 依次更新每个w
            for h in range(len(w_arr)):
                w_arr[h] = w_arr[h] - lr * w_grad[h]/np.sqrt(w_sum_arr[h])
            w_his.append(w_arr)

        print(b)
        print(w_arr)
        return b, w_arr, b_his, w_his

