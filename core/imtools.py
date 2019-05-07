
import os
import tensorflow as tf
import math
import numpy as np
import cv2

def imread(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError('I wonder if there is such a file: %s'%path)
    assert(image.dtype == np.uint8)
    normalized = np.asarray(image, dtype=np.float32)/255
    assert(len(normalized.shape) == 3)
    decolorized = np.mean(normalized, 2)
    return decolorized

def imsave(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.asarray(image*255., dtype=np.uint8)
    cv2.imwrite(path, data)

def shrink(img, factor):
    assert(isinstance(factor, int))
    embiggen = 1.0/factor
    return cv2.resize(img, (0, 0), fx=embiggen, fy=embiggen, interpolation=cv2.INTER_NEAREST)

def evaluate_single(image):
    return image.eval()[0, ..., 0]

def to_tf_image_shape(image_2d):
    return tf.reshape(image_2d, [1]+list(image_2d.shape)+[1])

def plot_image(image, title, ax):
    ax.imshow(image, cmap='gray', clim=(0, 1), interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')