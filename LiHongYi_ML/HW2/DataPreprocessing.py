# -*- coding: utf-8 -*-
# @Time    : 2019/4/27 16:14
# @Author  : Chen Jinghan
# @File    : DataPreprocessing.py
import pandas as pd


class DataPreprocessing:
    def __init__(self):
        pass

    def get_data(self):
        data = pd.read_csv('train.csv')

        print(data[:10])
        return data


if __name__ == '__main__':
    obj = DataPreprocessing()
    obj.get_data()

