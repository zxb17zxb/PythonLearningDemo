#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
"""
蘑菇中毒数据集,进行有毒-无毒的二分类
利用该数据集进行t-SNE特征可视化，并与PCA降维的结果进行对比
"""

filePath = r'E:\PythonProject\t-SNE\archive'

df = pd.read_csv(filePath + '\\mushrooms.csv')
print(df.head())

X = df.drop('class', axis = 1)  # 去除class 列
y = df['class']
y = y.map({'p': 'Posionous', 'e': 'Edible'})

cat_cols = X.select_dtypes(include='object').columns.tolist() #提取了名称
# print(cat_cols)
for col in cat_cols:
    print(f"col name: {col}, N Unique : {X[col].nunique()}")# 分别统计每一列属性各自有多少个不同值。

for col in cat_cols:
    X[col]=X[col].astype("category") #astype()函数可用于转化dateframe某一列的数据类型
    X[col]=X[col].cat.codes
print(X.head())

# """
# 使用PCA的降维可视化
# """
X_std = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_std)
print(X_pca.shape)
X_pca = np.vstack((X_pca.T, y)).T   # .T  转置  np.vstack 数值方向进行堆叠

df_pca = pd.DataFrame(X_pca, columns=['1st_Component', '2nd_Component', 'class'])
print("569887754645")
print(df_pca.head())

plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_pca, hue='class', x="1st_Component", y="2nd_Component")
plt.show()

"""
t-SNE 
"""
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_std)
X_tsne_data = np.vstack((X_tsne.T, y)).T
df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
print(df_tsne.head())

plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
plt.show()