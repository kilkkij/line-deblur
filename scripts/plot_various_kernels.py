
from typing import List
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
sys.path.append('../core')
from kernels import normal_spread_line_kernel
from imtools import plot_image

tf.InteractiveSession()

def kernel_from_vector(params: List[float]):
    return normal_spread_line_kernel(tf.constant(np.asarray(params, dtype=np.float32)))

plot_image(kernel_from_vector([2., 4.]).eval()[..., 0, 0], '[2., 4.]', plt.subplots()[1])
plot_image(kernel_from_vector([4., 2.]).eval()[..., 0, 0], '[4., 2.]', plt.subplots()[1])
plot_image(kernel_from_vector([-4., 1.]).eval()[..., 0, 0], '[-4., 1.]', plt.subplots()[1])
plot_image(kernel_from_vector([-10., 4.]).eval()[..., 0, 0], '[-10., 4.]', plt.subplots()[1])
plot_image(kernel_from_vector([-15., 8.]).eval()[..., 0, 0], '[-15., 8.]', plt.subplots()[1])
plot_image(kernel_from_vector([1., 0.]).eval()[..., 0, 0], '[1., 0.]', plt.subplots()[1])

plt.show()