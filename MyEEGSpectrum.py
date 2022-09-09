#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# filter data and resample

import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy import signal
from alphawaves.dataset import AlphaWaves
import numpy as np

def loadtxtmethod(filename):
    data = np.loadtxt(filename, dtype=np.float32, delimiter=',')
    return data


# 读入采集得到的EEG原始信号
fileOpen = 'F:\\1-临时文件\\8-9\\有明显效果的对照组\\16-好\\20220811140337661.txt'
fileClose = 'F:\\1-临时文件\\8-9\\有明显效果的对照组\\16-好\\20220811140436374.txt'
raw_open = loadtxtmethod(fileOpen)
raw_close = loadtxtmethod(fileClose)
#计算功率谱PSD
f, S_opened = signal.welch(raw_open,fs=250,axis=-1)
f, S_closed = signal.welch(raw_close,fs=250,axis=-1)
# 绘制原始信号的波形
# plot the results
fig = plt.figure(facecolor='white', figsize=(8, 6))
plt.plot(f, S_closed, c='k', lw=4.0, label='closed')
plt.plot(f, S_opened, c='r', lw=4.0, label='open')
plt.xlim(0, 40)
plt.xlabel('frequency', fontsize=14)
plt.title('PSD on both conditions(pwelch method)', fontsize=16)
plt.legend()
plt.show()

