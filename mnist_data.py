import tensorflow as tf
from tensorflow import keras
import keras.datasets.mnist as mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)


class MNIST():

    def __init__(self):
        pass

    def data():

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x = np.concatenate([x_train, x_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        x = x / 255.
        x = np.expand_dims(x, -1)

        return x, y
