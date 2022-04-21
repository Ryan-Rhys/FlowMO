# Author: Henry Moss & Ryan-Rhys Griffiths
"""
Utility methods for graph-based kernels
"""

import tensorflow as tf


def normalize(k_matrix):
    k_matrix_diagonal = tf.linalg.diag_part(k_matrix)
    squared_normalization_factor = tf.multiply(tf.expand_dims(k_matrix_diagonal, 1),
                                               tf.expand_dims(k_matrix_diagonal, 0))

    return tf.divide(k_matrix, tf.sqrt(squared_normalization_factor))
