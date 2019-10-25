from pathlib import Path
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import keras
from numpy.random import *
from PIL import Image
from keras.models import load_model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


'''画像とラベルの取り込み'''
train_folder = "/Users/sin/Documents/Workspace/ConvolutionNet/dataset/braille_train"
test_folder = "/Users/sin/Documents/Workspace/ConvolutionNet/dataset/braille_test"
validation_folder = "/Users/sin/Documents/Workspace/ConvolutionNet/dataset/braille_val"


'''ネットワーク'''
def network(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(40, kernel_size=16, padding="same", input_shape=input_shape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(60, kernel_size=8, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Dense(256))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    return model

'''データセット'''
class BrailleDataset():

    def __init__(self):
        self.image_shape = (32, 32, 1)
        self.num_classes = 10

    def crop_resize(self, image_path):
        image = Image.open(image_path).convert('L')

        resized = image.resize(self.image_shape[:2])
        img = np.array(resized).astype("float32")
        img /= 255
        return img

    def get_train_dataset(self):
        folder = Path(train_folder)
        image_paths = [str(f) for f in folder.glob("*.png")]

        '''画像の取り込み'''
        images = [self.crop_resize(p) for p in image_paths]
        images = np.asarray(images)

        image_names = []
        label = []
        '''ラベル名の取り込み'''
        image_names = [os.path.basename(i) for i in image_paths]
        for f in image_names:
            label.append(f.split('_')[0])

        return images, label

    def get_test_dataset(self):
        folder = Path(test_folder)
        image_paths = [str(f) for f in folder.glob("*.png")]

        '''画像の取り込み'''
        images = [self.crop_resize(p) for p in image_paths]
        images = np.asarray(images)

        image_names = []
        label = []
        '''ラベル名の取り込み'''
        image_names = [os.path.basename(i) for i in image_paths]
        for f in image_names:
            label.append(f.split('_')[0])

        return images, label

    def get_val_dataset(self):
        folder = Path(validation_folder)
        image_paths = [str(f) for f in folder.glob("*.png")]

        '''画像の取り込み'''
        images = [self.crop_resize(p) for p in image_paths]
        images = np.asarray(images)

        image_names = []
        label = []
        '''ラベル名の取り込み'''
        image_names = [os.path.basename(i) for i in image_paths]
        for f in image_names:
            label.append(f.split('_')[0])

        return images, label

    def get_batch(self):
        (x_train, y_train) = self.get_train_dataset()
        (x_test, y_test) = self.get_test_dataset()
        (x_val, y_val) = self.get_val_dataset()

        x_train, x_test, x_val = [self.preprocess(d) for d in [x_train, x_test, x_val]]
        y_train, y_test, y_val = [self.preprocess(d, label_data=True) for d in [y_train, y_test, y_val]]

        return x_train, y_train, x_test, y_test, x_val, y_val

    def preprocess(self, data, label_data=False):
        if label_data:
            data = keras.utils.to_categorical(data, self.num_classes)
        else:
            data = data.astype("float32")
            data /= 255
            shape = (data.shape[0],) + self.image_shape
            data = data.reshape(shape)

        return data

class Trainer():

    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        self.verbose = 1
        self.log_dir = os.path.join(os.path.dirname(__file__), "logdir_braille")
        self.model_file_name = "model_file_densetwice.hdf5"

    def train(self, x_train, y_train, batch_size, epochs, validation_data):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)

        datagen = ImageDataGenerator(
            rotation_range=2
        )

        datagen.fit(x_train)



        self._target.fit(
            x_train, y_train, batch_size=batch_size,
            epochs=epochs,
            validation_data = validation_data,
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(os.path.join(self.log_dir, self.model_file_name), save_best_only=True)
            ],
            verbose=self.verbose,
        )

dataset = BrailleDataset()

model = network(dataset.image_shape, dataset.num_classes)

x_train, y_train, x_test, y_test, x_val, y_val = dataset.get_batch()

print(x_train.shape)
print(y_train.shape)
print(y_val.shape)

trainer = Trainer(model, loss="categorical_crossentropy", optimizer=Adam(lr=0.00005))
trainer.train(x_train, y_train, batch_size=20, epochs=60, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
print(score)





