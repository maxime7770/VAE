from zipfile import ZIP_MAX_COMMENT
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from layers import Sampling
from model import modelVAE


latent_dim = 2   # chose 2 for a representation in 2D
loss_weights = [1, 0.001]


def Encoder():
    '''
    Defines the encoder part of the network
    '''

    inputs = keras.Input(shape=(28, 28, 1))

    # 32 3*3 kernels, with stride 1 and padding 'same' so the output is 32*28*28
    x = layers.Conv2D(32, 3, strides=1, padding='same',
                      activation='relu')(inputs)

    # 64 kernels, with stride 2 and padding 'same' so the output is 64*14*14
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)

    # output will be 7*7*64
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)

    # output will be 7*7*64 again because of stride 1
    x = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(x)

    x = layers.Flatten()(x)

    x = layers.Dense(16, activation='relu')(x)

    # z_mean and z_log_var are trained to be mean and log_var, shape batch_size*latent_dim
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    # Now generate a z with respect to the normal distribution given by z_mean and z_log_var

    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.compile()

    return encoder


def Decoder():
    '''
    defines the decoder part of the network
    '''

    inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(7*7*64,  activation='relu')(inputs)

    x = layers.Reshape((7, 7, 64))(x)
    # now we have 3 dimensions, we use transposed convolutions to get 28*28*1 dimension

    # the output will be 64*7*7
    x = layers.Conv2DTranspose(
        64, 3, strides=1, padding='same', activation='relu')(x)

    # stride 2 with padding 'same' double the size: output is 64*14*14
    x = layers.Conv2DTranspose(
        64, 3, strides=2, padding='same', activation='relu')(x)

    # double again : output is 32*28*28
    x = layers.Conv2DTranspose(
        32, 3, strides=2, padding='same', activation='relu')(x)

    # 1 filter so the output will be 1*28*28
    outputs = layers.Conv2DTranspose(
        1, 3, strides=1, padding='same', activation='sigmoid')(x)

    decoder = keras.Model(inputs, outputs, name='decoder')
    decoder.compile()

    return decoder


VAE = modelVAE(Encoder(), Decoder(), loss_weights)

VAE.compile(optimizer='adam')
