#from pathlib import Path
#import os
#import pathlib
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#import numpy as np
from keras.utils import plot_model

#path = "logdir_braille/model_file_densetwice.hdf5"
path = "../dataset/braille_train"

model = load_model(path)

lays = model.layers
for i, l in enumerate(lays):
    print(i+1, l)

w = model.layers[2].get_weights()[0]

w.transpose(3, 2, 0, 1)
nb_filter, nb_channel, nb_row, nb_col = w.shape
print(w.shape)

plt.figure()
for i in range(nb_col):
    im = w[:, :, 0, i]

    plt.subplot(6, 10, i + 1)
    plt.axis('off')
    plt.imshow(im, cmap="gray")
plt.show()