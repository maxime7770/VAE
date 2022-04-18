import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class Sampling(layers.Layer):

    def call(self, inputs):
        '''
        Build the Sampling layer given the inputs
        It is simply generating z given mean and variance
        '''

        z_mean, z_log_var = inputs

        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]

        eps = tf.keras.backend.random_normal(shape=(batch_size, latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps
