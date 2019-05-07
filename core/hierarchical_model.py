import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import namedtuple

import kernels

tfd = tfp.distributions
PosteriorComponents = namedtuple('PosteriorComponents', 'log_posterior log_likelihood log_prior kernel')

def log_posterior(obs, image, kernel_params, diff_stdev, error_stdev):
    prior = log_prior(image, kernel_params, diff_stdev)
    kernel = kernels.normal_spread_line_kernel(kernel_params)
    likelihood = log_likelihood(obs, image, kernel, error_stdev)
    return PosteriorComponents(
        likelihood + prior,
        likelihood,
        prior,
        kernel)

def cropped(image):
    kernel_pad = kernels.MAX_SIZE//2
    return image[:, kernel_pad:-kernel_pad, kernel_pad:-kernel_pad, :]

def log_likelihood(obs, image, kernel, error_stdev):
    convolved = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "SAME")
    diff = cropped(obs) - cropped(convolved)
    log_probs = tfd.Normal(0., scale=error_stdev).log_prob(diff)
    return tf.math.reduce_mean(log_probs)

def dual_normal_mixture(prob1, scale1, scale2):
    return tfd.Mixture(
        cat=tfd.Categorical(probs=[prob1, 1.-prob1]),
        components=[
            tfd.Normal(0.0, scale=scale1),
            tfd.Cauchy(0.0, scale=scale2),
    ])

def log_prior(image, kernel_params, diff_stdev):
    dy, dx = tf.image.image_gradients(image)
    zeroth_order_prior = tf.math.reduce_mean(tfd.Normal(.5, scale=.5).log_prob(image))
    flat_probability = 0.10
    first_order_distr = dual_normal_mixture(flat_probability, 0.03, diff_stdev)
    first_order_prior = tf.math.reduce_mean(first_order_distr.log_prob(dy)) + tf.math.reduce_mean(first_order_distr.log_prob(dx))
    first_order_stdev_prior = tfd.Gamma(3., 50.).log_prob(tf.math.square(diff_stdev))
    laplacian = tf.nn.conv2d(image, kernels.laplace_kernel(), [1, 1, 1, 1], "SAME")
    second_order_prior = tf.reduce_mean(tfd.Normal(0, scale=0.05).log_prob(laplacian))
    kernel_size_prior = tfd.Gamma(2., .02).log_prob(tf.reduce_sum(tf.square(kernel_params)))
    return zeroth_order_prior + first_order_prior + first_order_stdev_prior + kernel_size_prior + second_order_prior
