import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from mnist_data import MNIST
from layers import Sampling
import scipy
import scipy.stats

encoder = load_model('encoder.h5', custom_objects={'Sampling': Sampling})
decoder = load_model('decoder.h5')

x, y = MNIST.data()

x_show = x[0]

plt.imshow(x_show, cmap='binary')
plt.show()

z_mean, z_var, z = encoder.predict(np.array([x_show]))
x_reconst = decoder.predict(z)

x_reconst = np.squeeze(x_reconst[0], -1)

plt.imshow(x_reconst, cmap='binary')
plt.show()


def pick_dataset(*data, n=5):
    '''Return random subsets of n elements'''
    ii = np.random.choice(range(len(data[0])), n)
    out = [d[ii] for d in data]
    return out[0] if len(out) == 1 else out


n_show = 20000

# ---- Select images

x_show, y_show = pick_dataset(x, y, n=n_show)

# ---- Get latent points

z_mean, z_var, z = encoder.predict(x_show)

# ---- Show them

fig = plt.figure(figsize=(14, 10))
plt.scatter(z[:, 0], z[:, 1], c=y_show, cmap='tab10', alpha=0.5, s=30)
plt.colorbar()
plt.show()


grid_size = 18
grid_scale = 1

# ---- Draw a ppf grid

grid = []
for y in scipy.stats.norm.ppf(np.linspace(0.99, 0.01, grid_size), scale=grid_scale):
    for x in scipy.stats.norm.ppf(np.linspace(0.01, 0.99, grid_size), scale=grid_scale):
        grid.append((x, y))
grid = np.array(grid)

# ---- Draw latentspoints and grid

fig = plt.figure(figsize=(10, 8))
plt.scatter(z[:, 0], z[:, 1], c=y_show, cmap='tab10', alpha=0.5, s=20)
plt.scatter(grid[:, 0], grid[:, 1], c='black',
            s=60, linewidth=2, marker='+', alpha=1)
plt.show()

# ---- Plot grid corresponding images


def plot_images(x, y=None, indices='all', columns=12, x_size=1, y_size=1,
                colorbar=False, y_pred=None, cm='binary', norm=None, y_padding=0.35, spines_alpha=1,
                fontsize=20, interpolation='lanczos', save_as='auto'):
    """
    Show some images in a grid, with legends
    args:
        x             : images - Shapes must be (-1,lx,ly) (-1,lx,ly,1) or (-1,lx,ly,3)
        y             : real classes or labels or None (None)
        indices       : indices of images to show or 'all' for all ('all')
        columns       : number of columns (12)
        x_size,y_size : figure size (1), (1)
        colorbar      : show colorbar (False)
        y_pred        : predicted classes (None)
        cm            : Matplotlib color map (binary)
        norm          : Matplotlib imshow normalization (None)
        y_padding     : Padding / rows (0.35)
        spines_alpha  : Spines alpha (1.)
        font_size     : Font size in px (20)
        save_as       : Filename to use if save figs is enable ('auto')
    returns: 
        nothing
    """
    if indices == 'all':
        indices = range(len(x))
    if norm and len(norm) == 2:
        norm = matplotlib.colors.Normalize(vmin=norm[0], vmax=norm[1])
    draw_labels = (y is not None)
    draw_pred = (y_pred is not None)
    rows = math.ceil(len(indices)/columns)
    fig = plt.figure(figsize=(columns*x_size, rows*(y_size+y_padding)))
    n = 1
    for i in indices:
        axs = fig.add_subplot(rows, columns, n)
        n += 1
        # ---- Shape is (lx,ly)
        if len(x[i].shape) == 2:
            xx = x[i]
        # ---- Shape is (lx,ly,n)
        if len(x[i].shape) == 3:
            (lx, ly, lz) = x[i].shape
            if lz == 1:
                xx = x[i].reshape(lx, ly)
            else:
                xx = x[i]
        img = axs.imshow(xx,   cmap=cm, norm=norm, interpolation=interpolation)
#         img=axs.imshow(xx,   cmap = cm, interpolation=interpolation)
        axs.spines['right'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.spines['right'].set_alpha(spines_alpha)
        axs.spines['left'].set_alpha(spines_alpha)
        axs.spines['top'].set_alpha(spines_alpha)
        axs.spines['bottom'].set_alpha(spines_alpha)
        axs.set_yticks([])
        axs.set_xticks([])
        if draw_labels and not draw_pred:
            axs.set_xlabel(y[i], fontsize=fontsize)
        if draw_labels and draw_pred:
            if y[i] != y_pred[i]:
                axs.set_xlabel(f'{y_pred[i]} ({y[i]})', fontsize=fontsize)
                axs.xaxis.label.set_color('red')
            else:
                axs.set_xlabel(y[i], fontsize=fontsize)
        if colorbar:
            fig.colorbar(img, orientation="vertical", shrink=0.65)
    plt.show()


x_reconst = decoder.predict([grid])
plot_images(x_reconst, indices='all', columns=grid_size, x_size=0.5,
            y_size=0.5, y_padding=0, spines_alpha=0.1, save_as='09-Latent-morphing')
