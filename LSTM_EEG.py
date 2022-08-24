#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 导入工具包
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import SimpleRNN, LSTM, GRU
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Nadam, Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import np_utils
import scipy.io
import numpy as np
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras_visualizer import visualizer
import matplotlib.pyplot as plt
import time

fileName = 'E:\\1-毕业课题项目\\拿来练手的EEG项目\\LSTM_EEG\LSTM_EEG_Maosen-master\\LSTM_EEG_Maosen-master\\'
data = scipy.io.loadmat(fileName+'DATA\sp1s_aa_1000Hz.mat') # 'dict' object
# data = scipy.io.loadmat('DATA\sp1s_aa.mat')
y_test = np.loadtxt(fileName+'DATA\labels of the test set.txt',encoding="utf-8")

print("len(data):", len(data))
print(data)
print("len(y_test):", len(y_test))

x_train = data['x_train'].reshape((316, 500, 28))
# x_train /= 200
x_train = x_train.astype('float32')
"""
# 316个训练集样本
# 将训练数据调整为LSTM的正确输入尺寸并将数据转换为float 32
# 这里为什么要除以200？？
"""
x_test = data['x_test'].reshape((100, 500, 28))
# 100个测试集样本
# x_test /= 200
x_test = x_test.astype('float32')

# 标签数据
# 将标签数据调整为LSTM的正确输入尺寸，并将数据转化为float 32
# 这是一个标签
y_train = data['y_train'].reshape(316, 1)
# 标签1还是1，0转化为-1
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
# visualizer(model,format='png',view=True)

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
# toCallBack = TensorBoard(log_dir='./Graph',histogram_freq=1,write_graph=True,write_images=True)

class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 按照batch来进行追加数据
    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        # 每五秒按照当前容器里的值来绘图
        if int(time.time()) % 5 == 0:
            self.draw_p(self.losses['batch'], 'loss', 'train_batch')
            self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
            self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
            self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')

    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        # 每五秒按照当前容器里的值来绘图
        if int(time.time()) % 5 == 0:
            self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
            self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
            self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
            self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

    # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label + '.jpg')

    # 由于这里的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-5秒的时间）
    # 所以这里的方法会在整个训练结束以后调用
    def end_draw(self):
        self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
        self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
        self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
        self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
        self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

logs_loss = LossHistory()

model.fit(x_train, y_train, epochs=15, batch_size=20,callbacks=[logs_loss])

loss, acc = model.evaluate(x_test, y_test,
                            batch_size=1)
print('测试得分:', loss)
print('测试精度:', acc)


