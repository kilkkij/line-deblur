
import tensorflow as tf
import imtools
import kernels

cases = ['lena', 'library']

for case in cases:
    true_kernel_params = tf.constant([2.0, 4.0])
    true_kernel = kernels.normal_spread_line_kernel(true_kernel_params)
    path = '../data/%s.jpg'%case
    raw_data = imtools.imread(path)
    latent_image_data = imtools.shrink(raw_data, 3)
    latent_image = imtools.to_tf_image_shape(tf.constant(latent_image_data, dtype=tf.float32))
    blurred = tf.nn.conv2d(latent_image, true_kernel, [1, 1, 1, 1], "SAME")
    obs = blurred + tf.random.normal(
        blurred.shape,
        mean=0.0,
        stddev=.005,
        seed=123,
        dtype=tf.dtypes.float32)

    with tf.Session() as session:
        obs_data = imtools.evaluate_single(obs)
        imtools.imsave(latent_image_data, '%s-latent.jpg'%case)
        imtools.imsave(obs_data, '%s-obs.jpg'%case)