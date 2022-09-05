from tensorflow import keras
import matplotlib.pyplot as plt

# 加载mnist数据集
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
from tensorflow import keras
import matplotlib.pyplot as plt
# 加载mnist数据集
'''
.shape的使用：
    print(img.shape)        # 返回图像的高度、宽度以及通道数
    print(img.shape[0])     # 元组的第一个元素为图片数量
    print(img.shape[1])     # 元组的第一列元素为维度
    print(img.shape[2])     # 元组的第二列元素为列数
'''
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
print("train_images info:", train_images.shape, train_images.shape[0], train_images.shape[1],train_images.shape[2])
print("train_labels info:", train_labels.shape, train_labels.shape[0])
print("test_images info:", test_images.shape, test_images.shape[0], test_images.shape[1],test_images.shape[2])
print("test_labels info:", test_labels.shape, test_labels.shape[0])
# 绘制训练集第一张图片
# print(train_images[0]) # 第一张图像矩阵
plt.imshow(train_images[1])
plt.show()
