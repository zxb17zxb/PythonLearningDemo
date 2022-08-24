#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 导入工具包
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import SimpleRNN, LSTM, GRU
from tensorflow.keras.optimizers import SGD, Nadam, Adam, RMSprop
from keras.callbacks import TensorBoard
from keras.utils import np_utils
import scipy.io
import numpy as np
fileName = 'E:\\1-毕业课题项目\\拿来练手的EEG项目\\LSTM_EEG\LSTM_EEG_Maosen-master\\LSTM_EEG_Maosen-master\\'
data = scipy.io.loadmat(fileName+'DATA\sp1s_aa_1000Hz.mat') # 'dict' object
# data = scipy.io.loadmat('DATA\sp1s_aa.mat')
y_test = np.loadtxt(fileName+'DATA\labels of the test set.txt',encoding="utf-8")

print("len(data):",len(data))
print(data)
print("len(y_test):",len(y_test))

"""
将训练数据调整为LSTM的正确输入尺寸
并将数据转换为float 32
"""
x_train = data['x_train'].reshape((316,500,28))
x_train /= 200
x_train = x_train.astype('float32')
"""
将测试数据调整为LSTM的正确输入尺寸
并将数据转换为float 32
"""
x_test = data['x_test'].reshape((100,500,28))
x_test /= 200
x_test = x_test.astype('float32')

"""
将标签数据调整为LSTM的正确输入尺寸
并将数据转换为float 32
"""
y_train = data['y_train'].reshape(316,1)
tmp_train = []
for i in y_train:
    if i == 1:
        tmp_train.append(1)
    elif i == 0:
        tmp_train.append(-1)
y_train = np.array(tmp_train)
y_train = np_utils.to_categorical(y_train, 2)
y_train = y_train.astype('float32')

y_test = y_test.reshape(100,1)
tmp_test = []
for i in y_test:
    if i == 1:
        tmp_test.append(1)
    elif i == 0:
        tmp_test.append(-1)
y_test = np.array(tmp_test)
y_test = np_utils.to_categorical(y_test, 2)
y_test = y_test.astype('float32')

model = Sequential()
model.add(LSTM(10, return_sequences = True, input_shape=(500, 28)))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(5))
model.add(Dense(2, activation = 'softmax'))

model.summary()

"""
优化器设置
学习率为0.001
"""
optim = Nadam(lr = 0.001)
# 设置损失函数为交叉熵损失函数
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])

"""

epochs设置为10
batch_size设置为20

"""

model.fit(x_train, y_train, epochs=15, batch_size=20)

score, acc = model.evaluate(x_test, y_test,
                            batch_size=1)
print('测试得分:', score)
print('测试精度:', acc)


