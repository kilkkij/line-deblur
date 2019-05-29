
import tensorflow as tf
from core.kernels import line_kernel, MAX_SIZE

class TestLineKernel(tf.test.TestCase):

    def test_diagonal_kernel(self):
        kernel = line_kernel(tf.constant([3.5, 3.5]))
        middle = MAX_SIZE//2
        max_value = tf.reduce_max(kernel)
        self.assertAllClose(max_value, kernel[middle, middle, 0, 0])
        self.assertAllClose(max_value, kernel[middle+1, middle+1, 0, 0])
        side_value = 0.0549515038728714
        self.assertAllClose(tf.constant(side_value), kernel[middle-1, middle, 0, 0])
        self.assertAllClose(tf.constant(side_value), kernel[middle, middle-1, 0, 0])
        self.assertAllClose(tf.constant(0.), kernel[middle+1, middle-1, 0, 0])
        edge_value = 0.027475813403725624
        self.assertAllClose(tf.constant(edge_value), kernel[middle+2, middle+2, 0, 0])
        self.assertAllGreater(max_value, edge_value)

if __name__ == '__main__':
    tf.test.main()