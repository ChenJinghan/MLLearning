# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 17:14
# @Author  : Chen Jinghan
# @File    : DataPreprocessing.py
import random
import numpy as np
import pandas as pd


class DataPreprocessing:
    def __init__(self):
        pass

    def get_training_data(self):
        """
        get training data
        原始数据中一共给出18维的‘feature’
        tips：
        注意validation data的合理划分
        注意合理利用所有数据
        :return:预处理后的dataframe
        """
        # 数据读取与列命名
        train_dataframe = pd.read_csv('train.csv', header=0, dtype='str',
                                      names=['Date', 'city', 'item', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7',
                                             'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18',
                                             'h19', 'h20', 'h21', 'h22', 'h23'])

        # rainfall 中NR转0
        index = (train_dataframe.item == 'RAINFALL')
        rainfall = train_dataframe.loc[index]
        df = rainfall \
            .drop(columns='city') \
            .drop(columns='Date') \
            .drop(columns='item') \
            .applymap(lambda x: '0' if str(x) == 'NR' else str(x))

        train_dataframe.loc[
            index, ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13',
                    'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']] = df
        # dataframe 中所有str转double
        train_dataframe.loc[:, ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13',
                                'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']] \
            = train_dataframe.loc[:,
              ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13',
               'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']] \
            .applymap(lambda x: np.double(str(x)))
        # 认为‘city’列没用，删除
        train_dataframe = train_dataframe.drop(columns='city')

        return train_dataframe

    def get_feature_vector(self,df):
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
        # df_val = df.values
        # 按每天的feature循环，一天18个feature
        for i in range(0, (len(df) / 18)):
            # 按天循环，用前9天预测第10天的PM2.5
            for j in range(0, 15):
                df_lst.append(df.iloc[i * 18 + 9, j:j + 9])
                y_lst.append(df.iloc[i * 18 + 9, j + 9])
        return df_lst, y_lst

    def choose_validation_data(self, x_data, y_data):
        """
        从train.csv给出的数据中，划分training data和validation data
        :param x_data: feature list (每个元素均为df)
        :param y_data: label(每个元素均为具体值)
        :return: training data, training label, validation data, validation label
        """
        # 随机选出1500个作为validation的部分
        vali_index_set = set()
        while len(vali_index_set) < 1500:
            vali_index_set.add(random.randint(0, len(x_data) - 1))
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
            # 最后一个随机数已取
            if idx == len(vali_index_lst):
                for d in x_data[i:len(x_data)]:
                    train_data.append(d)
                for l in y_data[i:len(y_data)]:
                    train_label.append(l)
                break
            else:
                # 如果样本下标在随机数列表
                if i == vali_index_lst[idx]:
                    validate_data.append(x_data[i])
                    validate_label.append(y_data[i])
                    idx += 1
                # 如果样本下标不在随机数列表
                else:
                    train_data.append(x_data[i])
                    train_label.append(y_data[i])

        return train_data, train_label, validate_data, validate_label

    def get_testing_data(self):
        data = pd.read_csv('test.csv', header=None, dtype='str',
                        names=['ID','item', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8'])
        data = data.drop(columns='item').drop(columns='ID')
        df_lst = []
        for i in range(0, (len(data) / 18)):
            df_lst.append([np.double(i) for i in list(data.iloc[i * 18 + 9, :].values)])
        return df_lst

    def get_answer(self):
        ans = pd.read_csv('ans.csv',header=0,names=['id','value']).drop(columns='id')
        ans = [np.double(i) for i in ans.values.reshape(1,len(ans))[0]]
        return ans

if __name__ == '__main__':
    obj = DataPreprocessing()
    obj.get_answer()
