# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 21:07
# @Author  : Chen Jinghan
# @File    : PredictAndPaint.py
import numpy as np
import matplotlib.pyplot as plt

from LiHongYi_ML.HW1.Adagrad import Adagrad
from LiHongYi_ML.HW1.DataPreprocessing import DataPreprocessing
from LiHongYi_ML.HW1.GradientDescent import PredictByGradientDescent
from LiHongYi_ML.HW1.SGD import StochasticGradientDescent


def predict_on_validation_data(b, w_arr, validate_data):
    res_lst = []
    for i in range(0, len(validate_data)):
        res_lst.append(b + np.dot(w_arr, np.array(validate_data[i].values)))
    return res_lst


def mean_squared_error_on_validation(res_lst, validate_label):
    err = np.array(res_lst) - np.array(validate_label)
    err = [(i ** 2) for i in err]
    err = sum(err)/len(err)
    print("MSE on validation data: ", err)
    return err


def result_on_validation_data(res_lst, validate_label):
    print("res ", res_lst)
    print("y", validate_label)
    x = range(0, len(res_lst))
    plt.plot(x, res_lst, 'ro', color='r')
    plt.plot(x, validate_label, 'ro', color='g')
    plt.show()


def loss_paint(his_b, his_w, train_data, train_label,func):
    loss_lst = []

    # loss = (y-pre_y)的平方
    for i in range(len(his_b)):
        temp_lose = 0
        for j in range(len(train_data)):
            temp_lose += np.math.pow(train_label[j] - (his_b[i] + np.dot(his_w[i], train_data[j].values)), 2)

        loss_lst.append(temp_lose/len(train_data))
    x = range(len(his_b))
    y = loss_lst
    plt.plot(x, y, label=func)
    plt.title("training process")
    plt.xlabel("Iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(func + "loss.png")
    plt.show()


def predict_on_test_data(test_data, w_arr, b):
    res_lst = []
    res_lst.append(b + np.dot(test_data,w_arr))
    return res_lst[0]


def mse_on_test_data(res, ans):
    res = np.array(res)
    ans = np.array(ans)
    x = res - ans
    err = [i ** 2 for i in x]
    print("MSE on test data: ", sum(err) / len(err))


def main(func):
    pro = DataPreprocessing()
    train_dataframe = pro.get_training_data()
    df_lst, y_lst = pro.get_feature_vector(train_dataframe)
    train_data, train_label, validate_data, validate_label = pro.choose_validation_data(df_lst, y_lst)
    test_data = pro.get_testing_data()
    ans = pro.get_answer()

    if func == 'grd':
        grd = PredictByGradientDescent()
        b, w_arr, his_b, his_w = grd.gradient_descent(train_data, train_label)

        loss_paint(his_b, his_w, train_data, train_label, 'grd')
        res_lst = predict_on_validation_data(b, w_arr, validate_data)
        mean_squared_error_on_validation(res_lst, validate_label)
        res = predict_on_test_data(test_data, w_arr, b)
        mse_on_test_data(res=res, ans=ans)
        # result_on_validation_data(res_lst, validate_label)

    elif func == 'adagrad':
        ada = Adagrad()
        b, w_arr, his_b, his_w = ada.adagrad(train_data, train_label)

        loss_paint(his_b, his_w, train_data, train_label,'adagrad')
        res_lst = predict_on_validation_data(b, w_arr, validate_data)
        mean_squared_error_on_validation(res_lst, validate_label)
        res = predict_on_test_data(test_data, w_arr, b)
        mse_on_test_data(res=res, ans=ans)
        # result_on_validation_data(res_lst, validate_label)

    elif func == 'sgd':
        sgd = StochasticGradientDescent()
        b, w_arr, his_b, his_w = sgd.SDG(train_data, train_label)

        loss_paint(his_b, his_w, train_data, train_label,'sgd')
        res_lst = predict_on_validation_data(b, w_arr, validate_data)
        mean_squared_error_on_validation(res_lst, validate_label)
        res = predict_on_test_data(test_data, w_arr, b)
        mse_on_test_data(res=res, ans=ans)
        # result_on_validation_data(res_lst, validate_label)
    else:
        print("input error")


if __name__ == '__main__':
    main('sgd')
    # main('grd')
    # main('adagrad')


