# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 19:56
# @Author  : Chen Jinghan
# @File    : Adagrad.py

import numpy as np

from LiHongYi_ML.HW1.DataPreprocessing import DataPreprocessing


class Adagrad:
    def __init__(self):
        pass

    def adagrad(self, x_lst, y_lst):
        """
        using adagrad
        :param x_lst: traing data(consider PM2.5 only)
        :param y_lst: value of pm2.5
        :return: b, w_arr, b_his, w_his
        """
        # learning rate
        lr = 5
        iteration = 1000
        b = 0
        w_arr = np.zeros(len(x_lst[0]))
        b_sum = 0
        w_sum_arr = np.zeros(len(x_lst[0]))

        b_his = []
        w_his = []

        for i in range(iteration):
            b_grad = 0.0
            w_grad = np.zeros(len(x_lst[0]))
            # y=b+w1*x1+w2*x2+...+w9*x9
            # train by each training data
            for j in range(len(x_lst)):
                temp_feature = list(x_lst[j].values)
                b_grad = b_grad + 2 * (y_lst[j] - (b + np.dot(temp_feature, w_arr))) * (-1)
                # update each w
                for k in range(len(w_grad)):
                    print(w_grad)
                    w_grad[k] = w_grad[k] + 2 * (y_lst[j] - (b + np.dot(temp_feature, w_arr))) * (-temp_feature[k])

            b_sum += b_grad ** 2
            w_sum_arr += w_grad ** 2

            b = b - lr * b_grad/np.sqrt(b_sum)
            b_his.append(b)
            # update each w
            for h in range(len(w_arr)):
                w_arr[h] = w_arr[h] - lr * w_grad[h]/np.sqrt(w_sum_arr[h])
            w_his.append(w_arr)

        print("b:  ", b)
        print("w_arr:  ", w_arr)
        print("his_b", b)
        return b, w_arr, b_his, w_his

    def adagrad2(self, x_lst, y_lst):
        x_lst = [list(i.values) for i in x_lst]
        x_lst = np.array(x_lst)
        x_lst = np.hstack((x_lst, np.ones(len(x_lst)).reshape((len(x_lst),1))))

        # learning rate
        lr = 10
        iteration = 100
        w_arr = np.zeros(len(x_lst[0]))
        s_grad = np.zeros(len(x_lst[0]))
        for i in range(iteration):
            # y=b+w1*x1+w2*x2+...+w9*x9
            # 用每个training data 进行训练
            loss = y_lst - np.dot(x_lst, w_arr)
            grad = 2 * np.dot(np.array(x_lst).T, loss) * (-1)
            s_grad += grad ** 2
            w_arr -= lr * grad / np.sqrt(s_grad)
        return w_arr


if __name__ == '__main__':
    pro = DataPreprocessing()
    train_dataframe = pro.get_training_data()
    df_lst, y_lst = pro.get_feature_vector(train_dataframe)
    train_data, train_label, validate_data, validate_label = pro.choose_validation_data(df_lst, y_lst)

    ada = Adagrad()
    ada.adagrad2(train_data, train_label)
