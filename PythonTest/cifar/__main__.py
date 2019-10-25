import sys
import os
import pickle
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def unpickle(file):
    fp = open(file, 'rb')
    data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data

print('cwd = '+os.getcwd())
file = './data/cifar-10-batches-py/data_batch_1'

datadict = unpickle(file)
print(datadict.keys())
print(datadict['batch_label'])
print(datadict['labels'])
print(len(datadict['data'][0]))
print(datadict['data'][0])
print(datadict['filenames'][0])

label_names = unpickle("./data/cifar-10-batches-py/batches.meta")["label_names"]
data = datadict["data"]
labels = np.array(datadict["labels"])

# 各クラスの画像をランダムに10枚抽出して描画
nclasses = 10
pos = 1
for i in range(nclasses):
    # クラスiの画像のインデックスリストを取得
    targets = np.where(labels == i)[0]
    np.random.shuffle(targets)
    # 最初の10枚の画像を描画
    for idx in targets[:10]:
        plt.subplot(10, 10, pos)
        img = data[idx]
        # (channel, row, column) => (row, column, channel)
        plt.imshow(img.reshape(3, 32, 32).transpose(1, 2, 0))
        plt.axis('off')
        label = label_names[i]
        pos += 1
plt.show()