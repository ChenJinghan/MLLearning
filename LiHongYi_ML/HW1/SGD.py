# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 21:51
# @Author  : Chen Jinghan
# @File    : SGD.py
import numpy as np


class StochasticGradientDescent:
    def __init__(self):
        pass

    def SDG(self, x_lst, y_lst):
        """
        using Stochastic Gradient Descent
        not work don't know why.........
        :param x_lst: traing data(include data about PM2.5 only)
        :param y_lst: value of pm2.5
        :return: b, w_arr, b_his, w_his
        """
        lr = 0.001
        iteration = 100
        b = 0
        w_arr = np.zeros(len(x_lst[0]))
        his_b = []
        his_w = []

        for i in range(iteration):
            b_grad = 0.0
            w_grad = np.zeros(len(x_lst[0]))
            # update for each data
            for j in range(len(x_lst)):
                temp_feature = list(x_lst[j].values)
                b_grad = b_grad + 2 * (y_lst[j] - (b + np.dot(temp_feature,w_arr))) * (-1)
                print(b_grad)
                # update parameter b
                if (j+1) % 200 == 0:
                    b = b - lr * b_grad
                    his_b.append(b)
                    b_grad = 0.0
                # for each wi
                for k in range(len(w_grad)):
                    print(i,j,k)
                    w_grad[k] = w_grad[k] + 2 * (y_lst[j] - (b + np.dot(temp_feature,w_arr))) * (-temp_feature[k])

                    # update parameter w_arr
                    if (j+1) % 200 == 0:
                        w_arr[k] = w_arr[k] - lr * w_grad[k]
                if (j+1) % 200 == 0:
                    his_w.append(w_arr)
                    w_grad = np.zeros(len(x_lst[0]))

        print("b:  ", b)
        print("w_arr: ", w_arr)
        print("his_b: ", his_b)
        print("his_w: ", his_w)
        return b, w_arr, his_b, his_w
