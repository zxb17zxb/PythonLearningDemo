#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from Cython import inline
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
# %matplotlib inline

# 生成所有测试样本点
def make_meshgrid(x, y, h = .02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

#对测试集进行预测，并显示结果
def plot_test_results(ax, rf, xx, yy, **params):
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, **params)

#载入iris数据集
iris = datasets.load_iris()
#只使用前面两个特征向量
X = iris.data[:, :2]
#样本标签值
y = iris.target
#生成随机森林分类器，决策树最大深度为3，决策树数量为5
#决策树分裂时使用的特征数为1
rf = RandomForestClassifier(max_depth = 4, n_estimators = 5, max_features = 1)
#训练随机森林
rf.fit(X,y)
title = ('RandomForestClassifier')
# 设置绘图窗口
fig, ax = plt.subplots(figsize = (5, 5))
plt.subplots_adjust(wspace = 1, hspace = 1)
#前两个特征
X0, X1 = X[:, 0], X[:, 1]
#生成测试样本数据
xx, yy = make_meshgrid(X0, X1)
#对测试样本进行预测
plot_test_results(ax, rf, xx, yy, cmap = plt.cm.coolwarm, alpha = 0.8)
#显示训练样本
ax.scatter(X0, X1, c = y, cmap = plt.cm.coolwarm, s = 20, edgecolors = 'k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()