import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class modelVAE(keras.Model):
    '''
    Build a VAE model (both the encoder and the decoder)
    '''

    def __init__(self, encoder=None, decoder=None, loss_weights=[1, 1], **kwargs):
        '''
        VAE initialization:
            encoder: Encoder part of the model
            decoder: Decoder part of the model
            loss_weights: Weights of the loss functions (the reconstruction error and kl_loss)
        '''

        super(modelVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_weights = loss_weights

    def call(self, inputs):
        '''
        Take as input the inputs of the model 
        Return the output of the model
        '''

        z_mean, z_log_var, z = self.encoder(inputs)
        output = self.decoder(z)
        return output

    def train_step(self, input):
        ''' 
        One training step
        '''

        w1, w2 = self.loss_weights

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(input)
            reconstruction = self.decoder(z)

            reconstruction_loss = w1 * \
                tf.reduce_mean(
                    keras.losses.binary_crossentropy(input, reconstruction))

            # kl_loss is the KL divergence of the probability distribution computed by the encoder
            # and the normal distribution N(0,1)

            kl_loss = -w2 * \
                tf.reduce_mean(1 + z_log_var -
                               tf.square(z_mean) - tf.exp(z_log_var))

            loss = reconstruction_loss + kl_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"loss": loss, 'reconstruction_loss': reconstruction_loss, "kl_loss": kl_loss}

    def predict(self, inputs):
        '''
        Predict the ouptput given inputs
        '''
        z_mean, z_var_log, z = self.encoder.predict(inputs)
        outputs = self.decoder.predict(z)
        return outputs
