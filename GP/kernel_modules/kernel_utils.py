# Author: Henry Moss & Ryan-Rhys Griffiths
"""
Utility methods for graph-based kernels
"""

import tensorflow as tf
import numpy as np


def normalize(k_matrix):
    k_matrix_diagonal = tf.linalg.diag_part(k_matrix)
    squared_normalization_factor = tf.multiply(tf.expand_dims(k_matrix_diagonal, 1),
                                               tf.expand_dims(k_matrix_diagonal, 0))

    return tf.divide(k_matrix, tf.sqrt(squared_normalization_factor))


def pad_tensor(tensor, target_dim):
    return tf.pad(tensor, [[0, target_dim - tensor.shape[0]], [0, target_dim - tensor.shape[0]]], 'CONSTANT')


def pad_tensors(tensor_list):
    max_dim = max(tensor_list, key=lambda x: x.shape[0]).shape[0]
    return [pad_tensor(tensor, max_dim) for tensor in tensor_list]


def unpad_tensor(tensor):
    mask = tf.reduce_sum(tensor, 0) != 0
    rows_unpadded = tf.boolean_mask(tensor, mask, axis=0)
    fully_unpadded = tf.boolean_mask(rows_unpadded, mask, axis=1)
    return fully_unpadded


def preprocess_adjacency_matrix_inputs(adj_mat_list):
    padded_adj_mats = pad_tensors(adj_mat_list)
    flattened_padded_adj_mats = tf.reshape(padded_adj_mats, (len(padded_adj_mats), padded_adj_mats[0].shape[0]**2))
    return flattened_padded_adj_mats


def extract_adj_mats_from_vector_inputs(preprocessed_data):
    adj_mat_dim = int(np.sqrt(preprocessed_data.shape[1]))
    rehydrated_adj_mats = tf.reshape(preprocessed_data, (len(preprocessed_data), adj_mat_dim, adj_mat_dim))
    return rehydrated_adj_mats
