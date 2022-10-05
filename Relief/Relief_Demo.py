#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# An highlighted block
# _*_ coding:utf8 _*_

"""
# 说明：特征选择方法一：过滤式特征选择（Relief算法）
# 思想：先用特征选择过程对初始特征进行"过滤"，然后再用过滤后的特征训练模型
# 时间：2019-1-14
# 问题：
"""

import pandas as pd
import numpy as np
import numpy.linalg as la
import random


# 异常类
class FilterError:
    pass

# 第一步，定义类 和init方法
# 方便函数调用参数，只需要1次向类中导入参数即可，不用重复导入参数
class Filter:
    def __init__(self, data_df, sample_rate, t, k):
        """
        #
        :param data_df: 数据框（字段为特征，行为样本）
        :param sample_rate: 抽样比例
        :param t: 统计量分量阈值
        :param k: 选取的特征的个数
        """
        self.__data = data_df
        self.__feature = data_df.columns
        self.__sample_num = int(round(len(data_df) * sample_rate))
        self.__t = t
        self.__k = k

    # 数据处理（将离散型数据处理成连续型数据，比如字符到数值）
    # 将读取到的数据特征值中离散型处理成连续性，比如，色泽：青绿，属于离散型，密度：0。679， 属于连续型
    def get_data(self):
        new_data = pd.DataFrame() # 建立一个空二维表
        for one in self.__feature[:-1]: # 遍历循环每个特征
            col = self.__data[one] # 读取全部样本中其中一个特征
            # 判断读取到的特征是否全为数值类，如果字符串中全为数字，则不作改变写进新的二维表new_data里面，
            # 否则处理成数值类型写进二维表
            if (str(list(col)[0]).split(".")[0]).isdigit() or str(list(col)[0]).isdigit() \
                    or (str(list(col)[0]).split('-')[-1]).split(".")[-1].isdigit():#isdigit函数：如果是字符串包含数字返回ture,否则返回false
                new_data[one] = self.__data[one]
                # print '%s 是数值型' % one
            else:
                # print '%s 是离散型' % one
                keys = list(set(list(col))) #set函数：删除重复值,得到一个特征的类别，如色泽：青绿、浅白、乌黑
                values = list(range(len(keys))) #遍历循环len（keys)，色泽特征为例，三个特征类别，则得到三个数0、1、2
                new = dict(zip(keys, values)) #dict函数就是创建一个字典，zip函数矩阵中的元素对应打包成一个元组列表
                new_data[one] = self.__data[one].map(new) #map函数：将new: {'青绿': 0, '浅白': 1, '乌黑': 2}在col列表里做一个映射
        new_data[self.__feature[-1]] = self.__data[self.__feature[-1]] #瓜的类别属性不做改变
        return new_data

    # 返回一个样本的猜中近邻和猜错近邻
    # get_data(self)
    # 函数：经过处理数据，用0、1、2
    # 这样的数字去取代特征的中文类别，拿色泽为例，色泽 = ['青绿', '浅白', '乌黑']
    # 替换为
    # 色泽 = ['0', '1', '2']，完成所有替换后返回一个连续型数据类型的数据集。

    # 通过计算距离，找出猜错近邻和猜对近邻
    def get_neighbors(self, row):
        df = self.get_data()
        # row是一行一行（一个一个）样本
        row_type = row[df.columns[-1]]
        # 下面进行分类，与读取到的样本类型相同的分为  “同类”，不相同的分为 “异类”，储存在两个数据集中
        right_df = df[df[df.columns[-1]] == row_type].drop(columns=[df.columns[-1]])#筛选出数据集类别与读取样本同类的样本，删除数据集最后一列
        # 将删除后的数据集储存在right_df中，原数据集df保持不变。
        wrong_df = df[df[df.columns[-1]] != row_type].drop(columns=[df.columns[-1]])
        aim = row.drop(df.columns[-1])
        f = lambda x: eulidSim(np.mat(x.values), np.mat(aim.values))
        right_sim = right_df.apply(f, axis=1)
        right_sim_two = right_sim.drop(right_sim.idxmin())
        # print right_sim_two
        # print right_sim.values.argmax()   # np.argmax(wrong_sim)
        wrong_sim = wrong_df.apply(f, axis=1)
        # print wrong_sim
        # print wrong_sim.values.argmax()
        # print right_sim_two.idxmin(), wrong_sim.idxmin()
        return right_sim_two.idxmin(), wrong_sim.idxmin()

    # 计算特征权重
    def get_weight(self, feature, index, NearHit, NearMiss):
        data = self.__data.drop(self.__feature[-1], axis=1)
        row = data.iloc[index]
        nearhit = data.iloc[NearHit]
        nearmiss = data.iloc[NearMiss]
        if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
            max_feature = data[feature].max()
            min_feature = data[feature].min()
            right = pow(round(abs(row[feature] - nearhit[feature]) / (max_feature - min_feature), 2), 2)
            wrong = pow(round(abs(row[feature] - nearmiss[feature]) / (max_feature - min_feature), 2), 2)
            # w = wrong - right
        else:
            right = 0 if row[feature] == nearhit[feature] else 1
            wrong = 0 if row[feature] == nearmiss[feature] else 1
            # w = wrong - right
        w = wrong - right
        # print w
        return w

    # 过滤式特征选择
    def relief(self):
        sample = self.get_data()
        # print sample
        m, n = np.shape(self.__data)  # m为行数，n为列数
        score = []
        sample_index = random.sample(range(0, m), self.__sample_num)
        print('采样样本索引为 %s ' % sample_index)
        num = 1
        for i in sample_index:    # 采样次数
            one_score = dict()
            row = sample.iloc[i]
            NearHit, NearMiss = self.get_neighbors(row)
            print('第 %s 次采样，样本index为 %s，其NearHit行索引为 %s ，NearMiss行索引为 %s' % (num, i, NearHit, NearMiss))
            for f in self.__feature[0:-1]:
                w = self.get_weight(f, i, NearHit, NearMiss)
                one_score[f] = w
                print ('特征 %s 的权重为 %s.' % (f, w))
            score.append(one_score)
            num += 1
        f_w = pd.DataFrame(score)
        print ('采样各样本特征权重如下：')
        print (f_w)
        print ('平均特征权重如下：')
        print (f_w.mean())
        return (f_w.mean())

    # 返回最终选取的特征
    def get_final(self):
        f_w = pd.DataFrame(self.relief(), columns=['weight'])
        final_feature_t = f_w[f_w['weight'] > self.__t]
        print (final_feature_t)
        final_feature_k = f_w.sort_values('weight').head(self.__k)
        print (final_feature_k)
        return final_feature_t, final_feature_k


# 几种距离求解
def eulidSim(vecA, vecB):
    return la.norm(vecA - vecB)


def cosSim(vecA, vecB):
    """
    :param vecA: 行向量
    :param vecB: 行向量
    :return: 返回余弦相似度（范围在0-1之间）
    """
    num = float(vecA * vecB.T)
    denom = la.norm(vecA) * la.norm(vecB)
    cosSim = 0.5 + 0.5 * (num / denom)
    return cosSim


def pearsSim(vecA, vecB):
    if len(vecA) < 3:
        return 1.0
    else:
        return 0.5 + 0.5 * np.corrcoef(vecA, vecB, rowvar=0)[0][1]


if __name__ == '__main__':
    data = pd.read_csv('西瓜数据集3.csv')[['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率', '类别']]
    print(data)
    f = Filter(data, 1, 0.8, 6)
    f.relief()
    # f.get_final()


