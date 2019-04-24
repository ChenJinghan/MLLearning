# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 11:24
# @Author  : Chen Jinghan
# @File    : PredictePM2.5.py
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_training_data():
    """
    读取训练数据
    原始数据中一共给出18维的‘feature’
    tips：
    注意validation data的合理划分
    注意合理利用所有数据
    :return:预处理后的dataframe
    """
    # 数据读取与列命名
    train_dataframe = pd.read_csv('train.csv',header=0,dtype='str',
                                  names=['Date', 'city', 'item', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7',
                                         'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                                         'h19', 'h20', 'h21', 'h22', 'h23'])

    # rainfall 中NR转0
    index = (train_dataframe.item == 'RAINFALL')
    rainfall = train_dataframe.loc[index]
    df = rainfall\
        .drop(columns='city')\
        .drop(columns='Date')\
        .drop(columns='item')\
        .applymap(lambda x: '0' if str(x) == 'NR' else str(x))

    train_dataframe.loc[index, ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13',
                        'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']] = df
    # dataframe 中所有str转double
    train_dataframe.loc[:, ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13',
                            'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']]\
        = train_dataframe.loc[:, ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13',
                                  'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']]\
        .applymap(lambda x: np.double(str(x)))
    # 认为‘city’列没用，删除
    train_dataframe = train_dataframe.drop(columns='city')

    return train_dataframe


def get_feature_vector(df):
    """
    feature和训练数据的切分
    :param df: 预处理后的dataframe
    :return: featrue和label对应的两个list
    """
    # feature list
    df_lst = []
    # y list
    y_lst = []
    df = df.drop(columns='Date').drop(columns='item')
    # 按每天的feature循环，一天18个feature
    for i in range(0, (len(df)/18)):
        # 按天循环，用前9天预测第10天的PM2.5
        for j in range(0, 15):
            temp_df = df.iloc[i*18:(i+1)*18, j:j+9]
            df_lst.append(temp_df)
            # print([i*18+9 , j+9])
            y_lst.append(df.iloc[i*18+9, j+9])
    return df_lst, y_lst


def choose_validation_data(x_data, y_data):
    """
    从train.csv给出的数据中，划分training data和validation data
    :param x_data: feature
    :param y_data: label
    :return: training data, training label, validation data, validation label
    """
    # 随机选出1000个作为validation的部分
    vali_index_set = set()
    while len(vali_index_set) < 10:
        vali_index_set.add(random.randint(0,len(x_data)-1))
    vali_index_lst = list(vali_index_set)
    vali_index_lst.sort()
    # print(vali_index_lst)

    train_data = []
    train_label = []
    validate_data = []
    validate_label = []

    # 根据随机数进行划分
    idx = 0
    for i in range(0, len(x_data)):
        if idx == len(vali_index_lst):
            for d in x_data[i:len(x_data)]:
                train_data.append(d)
            for l in y_data[i:len(y_data)]:
                train_label.append(l)
            # train_data.append(d for d in x_data[i:len(x_data)])
            # train_label.append(l for l in y_data[i:len(y_data)])
            break
        else:
            if i == vali_index_lst[idx]:
                validate_data.append(x_data[i])
                validate_label.append(y_data[i])
                idx += 1
            else:
                train_data.append(x_data[i])
                train_label.append(y_data[i])

    # print("train_data: ", train_data)
    # print("train_label: ", train_label)
    # print("vali_data: ", validate_data)
    # print("vali_label: ", validate_label)
    return train_data, train_label, validate_data, validate_label


# y=b+w1*x1+w2*x2+...+w9*x9
def gradient_descent(x_lst, y_lst):
    """
    只考虑PM2.5预测PM2.5
    用Gradient Descent
    :return: bias 和 w
    """
    iteration = 25
    lr = 0.00000001
    b = 1.0
    w_arr = np.ones(9 * 1)

    for i in range(iteration):
        b_grad = 0.0
        w_grad = np.zeros(9 * 1)
        for i in range(0,len(x_lst)):
        # for i in range(1):
            temp_feature = x_lst[i].iloc[9].values
            b_grad = b_grad + 2 * (y_lst[i] - (b + np.dot(temp_feature, w_arr))) * (-1)
            for j in range(len(w_grad)):
                w_grad[j] = w_grad[j] + 2 * (y_lst[i] - (b + np.dot(temp_feature, w_arr))) * (-temp_feature[j])

        b = b - lr * b_grad
        for i in range(len(w_arr)):
            w_arr[i] = w_arr[i] - lr * w_grad[i]
    print(b)
    print(w_arr)
    return b, w_arr


def predict_on_validation_data(b, w_arr, validate_data):
    res_lst = []
    for i in range(0, len(validate_data)):
        res_lst.append(b + np.dot(w_arr, np.array(validate_data[i].iloc[9].values)))
    return res_lst


def result_on_validation_data(res_lst, validate_label):
    print("res ", res_lst)
    print("y", validate_label)
    x = range(0,len(res_lst))
    plt.plot(x, res_lst, 'ro', color = 'r')
    plt.plot(x, validate_label, 'ro', color = 'g')
    plt.show()


train_dataframe = get_training_data()
df_lst, y_lst = get_feature_vector(train_dataframe)
train_data, train_label, validate_data, validate_label = choose_validation_data(df_lst,y_lst)
b, w_arr = gradient_descent(train_data, train_label)
res_lst = predict_on_validation_data(b, w_arr, validate_data)
result_on_validation_data(res_lst, validate_label)

# choose_validation_data([0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8])