
import argparse
import tensorflow as tf

import sys
sys.path.append('../core')
import imtools
import kernels

parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str, help='input')
parser.add_argument('case_name', type=str, help='input')
parser.add_argument('--noise', dest='noise_stdev', type=str, default=0., help='noise standard deviation')
args = parser.parse_args()

true_kernel_params = tf.constant([2.0, 4.0])
true_kernel = kernels.line_kernel(true_kernel_params)
raw_data = imtools.imread(args.input_path)
latent_image_data = imtools.shrink(raw_data, 3)
latent_image = imtools.to_tf_image_shape(tf.constant(latent_image_data, dtype=tf.float32))
blurred = tf.nn.conv2d(latent_image, true_kernel, [1, 1, 1, 1], "SAME")
obs = blurred + tf.random.normal(
    blurred.shape,
    mean=0.0,
    stddev=args.noise_stdev,
    seed=123,
    dtype=tf.dtypes.float32)

with tf.Session() as session:
    obs_data = imtools.evaluate_single(obs)
    imtools.imsave(latent_image_data, 'cases/%s/latent.png'%args.case_name)
    imtools.imsave(obs_data, 'cases/%s/obs.png'%args.case_name)