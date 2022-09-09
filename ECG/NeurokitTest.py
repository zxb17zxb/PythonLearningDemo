#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import neurokit2 as nk
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob

"""
绘制庞加莱图等
"""


def HRV_Indice_Figure(raw, FigureOutputPath, file):
    data = raw['data'][:, 0]
    # Clean signal and find peaks
    ecg_cleaned = nk.ecg_clean(data, sampling_rate=2000)
    # Find peaks
    peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=2000)
    # Compute HRV indices 并绘图
    hrv_indices = nk.hrv(peaks, sampling_rate=2000, show=True)
    plt.savefig(FigureOutputPath + "\\" + file + ".jpg")
    return hrv_indices


"""
特征归一化函数
"""


def minmax(df):
    for column in df.columns:
        if (((df[column].dtype == np.int64) or (df[column].dtype == np.float_)) or (
                column == 'Valence' or column == 'Arousal' or column == 'Dominance') and not (
                column == "Video")):  # if not(df[column].dtype == np.object) and not(column=="Video"): # : Fails for integers
            if not ((np.max(df[column]) - np.min(df[column])) == 0):
                df[column] = (df[column] - np.min(df[column])) / (np.max(df[column]) - np.min(df[column]))
    return df


"""
特征提取
"""


def FeatureExtraction(data):
    # process with neurokit
    ecg_signals_b_l, info_b_l = nk.ecg_process(data, sampling_rate=2000)
    features_ecg_l = nk.ecg_intervalrelated(ecg_signals_b_l)
    # # 特征归一化 minmax 方法
    # df_Features = minmax(features_ecg_l)
    return features_ecg_l


"""
输出文件名
"""
output_filepath_feature = "F:\安装包2\RSA data\Feature_csv"
output_filename = "Feature_Extraction.csv"
FigureOutputPath = r"F:\安装包2\RSA data\Figure"
plt.rc('font', size=8)
"""
导入文件
"""
# 文件路径
FilePath = r"F:\安装包2\RSA data\RSA data"
# 获取多个文件夹的路径，并返回一个可迭代对象
files = os.listdir(FilePath)
print(files)

df_Features_all = pd.DataFrame()
df_Features_old = []
df_Features = pd.DataFrame()
i = 0
for file in files:
    try:
        used_name = FilePath + "\\" + file
        rawfile = used_name + "\\" + file + ".mat"
        print(rawfile)

        raw = sio.loadmat(rawfile)
        # hrv_indices = HRV_Indice_Figure(raw, FigureOutputPath, file)
        rawnew = raw['data'][:, 0]
        # rawnew = np.array(rawnew)
        # rawnew = rawnew.reshape(rawnew.shape[0], 1)
        # rawnew = rawnew[:100000]
        # rawnew = rawnew.reshape(rawnew.shape[0])

        df_Features_old = FeatureExtraction(rawnew)
        # 特征归一化 minmax 方法
        df_Features_old = minmax(df_Features_old)
        df_Features_old.insert(0, "participant", file)
        df_Features = df_Features.append(df_Features_old)
        print(df_Features)
        i += 1
    except:
        print(file + "出错")
        # empty = []
        #
        # for i in range(len(df_Features_old)):
        #     empty.append("empty")
        # df_Features_old.insert(i, "participants", file)
        # df_Features = df_Features.append(empty)
        # print(df_Features)
        # df_Features_old.insert()
        # i += 1
        pass

# 将提取的特征保存为.csv 文件
String = output_filepath_feature + "\\" + output_filename
# file = open(String, "w")
df_Features.to_csv(String)
