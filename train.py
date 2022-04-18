
import matplotlib.pyplot as plt
from build import VAE
from mnist_data import MNIST

epochs = 50
batch_size = 64

x, y = MNIST.data()

history = VAE.fit(x, epochs=epochs, batch_size=batch_size)


VAE.encoder.save('encoder.h5')
VAE.decoder.save('decoder.h5')


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
