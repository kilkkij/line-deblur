
import numpy as np
import tensorflow as tf
import unittest
import os
from src.line_kernel import normal_spread_line_kernel, MAX_SIZE

class TestLineKernel(tf.test.TestCase):

	def test_diagonal_kernel(self):
		kernel = normal_spread_line_kernel(tf.constant([3.5, 3.5]))
		middle = MAX_SIZE//2
		max_value = tf.reduce_max(kernel)
		self.assertAllClose(max_value, kernel[middle, middle, 0, 0])
		self.assertAllClose(max_value, kernel[middle+1, middle+1, 0, 0])
		side_value = 0.011107037775
		self.assertAllClose(tf.constant(side_value), kernel[middle-1, middle+1, 0, 0])
		self.assertAllClose(tf.constant(side_value), kernel[middle+1, middle-1, 0, 0])
		self.assertAllClose(tf.constant(0.), kernel[middle+3, middle-3, 0, 0])
		edge_value = 0.01942365989
		self.assertAllClose(tf.constant(edge_value), kernel[middle+2, middle+2, 0, 0])
		self.assertAllGreater(max_value, edge_value)

if __name__ == '__main__':
	tf.test.main()