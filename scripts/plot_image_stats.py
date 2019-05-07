
import tensorflow as tf
from pylab import *

import kernels
import imtools

import sys

path = sys.argv[1]

true_kernel_params = tf.constant([4.0, 2.0])
true_kernel = kernels.normal_spread_line_kernel(true_kernel_params)
latent_image_data = imtools.shrink(imtools.imread(path), 3)
latent_image = imtools.to_tf_image_shape(tf.constant(latent_image_data, dtype=tf.float32))

tf.InteractiveSession()

dy, dx = tf.image.image_gradients(latent_image)
fig, axes = subplots(1, 2, figsize=(10, 6))
imtools.plot_image(imtools.evaluate_single(dy), 'dy', axes[0])
imtools.plot_image(imtools.evaluate_single(dx), 'dx', axes[1])

diffs = np.concatenate([dy.eval()[0, ..., 0], dx.eval()[0, ..., 0]])
figure()
hist(diffs.flatten(), bins=300)

show()