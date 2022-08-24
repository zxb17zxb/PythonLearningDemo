#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 只有1个权重（w）的最小神经网络，我们给它输入一个输入（x），再乘以权重，结果就是网络的输出。
import keras
from keras.layers import Dense
import numpy as np

model = keras.models.Sequential()

model.add(Dense(units=1, use_bias=False, input_shape=(1,)))   # 仅有的1个权重在这里

model.compile(loss='mse', optimizer='adam')
"""
创建数据
"""
data_input = np.random.normal(size=100000)  # 训练数据

data_label = -(data_input)  # 数据标签
"""
训练网络
"""
# # 检查权重的大小
# model.layers[0].get_weights()
# # 将数据放入网络中
# model.fit(data_input, data_label)
# # 向网络提供一个值并检查响应
# model.predict(np.array([2.5]))
# # 检查训练成功网络后的权重值(W)。
# model.layers[0].get_weights()
print('模型随机权重分配为：%s' % (model.layers[0].get_weights()))  # 检查随机初始化的权重大小

model.fit(data_input, data_label, epochs=1, batch_size=1, verbose=1)  # 对创建的数据用创建的网络进行训练

print('模型进行预测：%s' % (model.predict(np.array([2.5]))))  # 利用训练好的模型进行预测

print('训练完成后权重分配为：%s' % (model.layers[0].get_weights()))  # 再次查看训练好的模型中的权重值