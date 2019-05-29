
from collections import OrderedDict
import numpy as np
import tensorflow as tf

import imtools
import model
import cost_logging

max_iterations = 3000
kernel_params_initial = [4.0, 2.0]
diff_stdev_initial = 0.03
error_stdev_initial = 0.01

def optimize(obs_data):
    with tf.Session():
        obs = imtools.to_tf_image_shape(tf.constant(obs_data))
        image = tf.Variable(obs)
        diff_stdev = tf.Variable(diff_stdev_initial)
        error_stdev = tf.Variable(error_stdev_initial)
        kernel_params = tf.Variable(np.array(kernel_params_initial, dtype=np.float32))
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1., rho=0.80)
        fast_optimizer = tf.train.AdadeltaOptimizer(learning_rate=50., rho=0.80)
        posterior_components = model.log_posterior(obs, image, kernel_params, diff_stdev, error_stdev)
        cost = -posterior_components.log_posterior
        steps = [
            optimizer.minimize(cost, var_list=[image]),
            fast_optimizer.minimize(cost, var_list=[kernel_params]),
            optimizer.minimize(cost, var_list=[diff_stdev]),
            optimizer.minimize(cost, var_list=[error_stdev])
            ]
        log = cost_logging.logger(OrderedDict([
            ('posterior', posterior_components.log_posterior),
            ('likelihood', posterior_components.log_likelihood),
            ('prior', posterior_components.log_prior),
            ('kernel p0', kernel_params[0]),
            ('kernel p1', kernel_params[1]),
            ('diff_stdev', diff_stdev),
            ('error_stdev', error_stdev)
            ]), interval=10)
        tf.initializers.global_variables().run()
        print('starting optimization, ctrl+c to cut short')
        try:
            for i in range(max_iterations):
                for step in steps:
                    step.run()
                log(i)
        except KeyboardInterrupt:
            pass
        return imtools.evaluate_single(image), posterior_components.kernel.eval()[..., 0, 0]