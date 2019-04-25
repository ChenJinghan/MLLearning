# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 21:07
# @Author  : Chen Jinghan
# @File    : PredictAndPaint.py
import numpy as np
import matplotlib.pyplot as plt

from LiHongYi_ML.HW1.Adagrad import Adagrad
from LiHongYi_ML.HW1.DataPreprocessing import DataPreprocessing
from LiHongYi_ML.HW1.GradientDescent import PredictByGradientDescent


def predict_on_validation_data(b, w_arr, validate_data):
    res_lst = []
    for i in range(0, len(validate_data)):
        res_lst.append(b + np.dot(w_arr, np.array(validate_data[i].iloc[9].values)))
    return res_lst


def result_on_validation_data(res_lst, validate_label):
    print("res ", res_lst)
    print("y", validate_label)
    x = range(0, len(res_lst))
    plt.plot(x, res_lst, 'ro', color='r')
    plt.plot(x, validate_label, 'ro', color='g')
    plt.show()


def loss_paint(his_b, his_w, train_data, train_label):
    loss_lst = []
    # loss = (y-pre_y)的平方
    for i in range(len(his_b)):
        loss_lst.append(np.math.pow(train_label[i] - (his_b[i] + np.dot(his_w[i], train_data[i].iloc[9].values)), 2))
    x = range(len((his_b)))
    y = loss_lst
    print("x: ", x)
    print("y: ", y)
    plt.plot(x, y)
    plt.show()


pro = DataPreprocessing()
train_dataframe = pro.get_training_data()
df_lst, y_lst = pro.get_feature_vector(train_dataframe)
train_data, train_label, validate_data, validate_label = pro.choose_validation_data(df_lst, y_lst)

grd = PredictByGradientDescent()
b, w_arr, his_b, his_w = grd.gradient_descent(train_data, train_label)

res_lst = predict_on_validation_data(b, w_arr, validate_data)
result_on_validation_data(res_lst, validate_label)
loss_paint(his_b, his_w, train_data, train_label)


# ada = Adagrad()
# ada_b, ada_w_arr, ada_b_his, ada_w_his = ada.adagrad(train_data, train_label)
#
# res_lst2 = predict_on_validation_data(ada_b, ada_w_arr, validate_data)
# result_on_validation_data(res_lst2, validate_label)
# loss_paint(ada_b_his, ada_w_his, train_data, train_label)

