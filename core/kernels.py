
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

MAX_SIZE = 10

def cross_product_2d(vector, X, Y):
    return vector[0]*Y - vector[1]*X

def normal_spread_line_kernel(params):
    x = np.arange(-MAX_SIZE/2, MAX_SIZE/2+1, dtype=np.float32)
    X, Y = tf.meshgrid(x, x)
    R2 = tf.math.square(X) + tf.math.square(Y)
    radius = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(params)))
    normalized_params = params/radius
    distance = cross_product_2d(normalized_params, X, Y)
    raw_line = tfp.distributions.Normal(loc=tf.zeros_like(distance), scale=0.635).prob(value=distance)
    clipped = _clipped_radius_cone(R2, radius)*raw_line
    normalized = clipped/tf.math.reduce_sum(clipped)
    in_standard_shape = tf.reshape(normalized, normalized.shape.as_list()+[1, 1])
    return in_standard_shape

def _clipped_radius_cone(R2, radius):
    return tf.clip_by_value(radius/2 - tf.math.sqrt(R2) + .5, 0., 1.)

def laplace_kernel():
    kernel_2d = tf.constant(np.asarray([
        [-.5, -1., -.5],
        [-1., 6., -1.],
        [-.5, -1., -.5]], dtype=np.float32)/6.)
    return tf.reshape(kernel_2d, kernel_2d.shape.as_list()+[1, 1])
