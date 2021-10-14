# Author: Henry Moss & Ryan-Rhys Griffiths
"""
Molecule kernels for Gaussian Process Regression implemented in GPflow.
"""

import gpflow
import numpy as np
import tensorflow as tf

from math import factorial

from .kernel_utils import normalize, unpad_tensor, extract_adj_mats_from_vector_inputs


class RandomWalk(gpflow.kernels.Kernel):
    def __init__(self, normalize=True, weight=0.1, series_type='geometric',  p=None, uniform_probabilities=False):
        super().__init__()
        self.normalize = normalize
        self.weight = weight
        if series_type == 'geometric':
            self.geometric = True
        elif series_type == 'exponential':
            self.geometric = False
        self.p = p
        self.uniform_probabilities = uniform_probabilities

    def K(self, X, X2=None):
        """
        Compute the random walk graph kernel (Gartner et al. 2003),
        specifically using the spectral decomposition approach
        given by https://www.jmlr.org/papers/volume11/vishwanathan10a/vishwanathan10a.pdf

        :param X: array of N graph objects (represented as adjacency matrices of varying sizes)
        :param X2: array of M graph objects (represented as adjacency matrices of varying sizes)
            If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M.
        """

        X_padded = False
        X2_padded = False
        X_is_X2 = False

        if isinstance(X, (np.ndarray, tf.Tensor)) and len(X.shape) == 2:
            X = extract_adj_mats_from_vector_inputs(X)
            X_padded = True

        if X2 is None:
            X2 = X
            X2_padded = X_padded
            X_is_X2 = True
        elif isinstance(X2, (np.ndarray, tf.Tensor)) and len(X2.shape) == 2:
            X2 = extract_adj_mats_from_vector_inputs(X2)
            X2_padded = True

        flattened_k_matrix = tf.TensorArray(tf.float64, size=len(X)*len(X2))
        matrix_idx = 0

        for idx_1 in range(len(X)):

            adj_mat_1 = X[idx_1]
            if X_padded:
                adj_mat_1 = unpad_tensor(adj_mat_1)
            eigenval_1, eigenvec_1 = tf.linalg.eigh(tf.cast(adj_mat_1, tf.float64))
            start_stop_probs = tf.ones((1, tf.shape(eigenvec_1)[0]), tf.float64)
            if self.uniform_probabilities:
                start_stop_probs = tf.divide(start_stop_probs, tf.shape(eigenvec_1)[0])
            flanking_factor_1 = tf.linalg.matmul(start_stop_probs, eigenvec_1)

            for idx_2 in range(len(X2)):

                if X_is_X2 and idx_1 == idx_2:
                    eigenval_2, eigenval_2, flanking_factor_2 = eigenval_1, eigenval_1, flanking_factor_1
                else:
                    adj_mat_2 = X2[idx_2]
                    if X2_padded:
                        adj_mat_2 = unpad_tensor(adj_mat_2)
                    eigenval_2, eigenvec_2 = tf.linalg.eigh(tf.cast(adj_mat_2, tf.float64))
                    start_stop_probs = tf.ones((1, tf.shape(eigenvec_2)[0]), tf.float64)
                    if self.uniform_probabilities:
                        start_stop_probs = tf.divide(start_stop_probs, tf.shape(eigenvec_2)[0])
                    flanking_factor_2 = tf.linalg.matmul(start_stop_probs, eigenvec_2)


                flanking_factor = tf.linalg.LinearOperatorKronecker(
                    [tf.linalg.LinearOperatorFullMatrix(flanking_factor_1),
                     tf.linalg.LinearOperatorFullMatrix(flanking_factor_2)
                     ]).to_dense()

                diagonal = self.weight * tf.linalg.LinearOperatorKronecker(
                    [tf.linalg.LinearOperatorFullMatrix(tf.expand_dims(eigenval_1, axis=0)),
                     tf.linalg.LinearOperatorFullMatrix(tf.expand_dims(eigenval_2, axis=0))
                     ]).to_dense()

                if self.p is not None:
                    power_series = tf.ones_like(diagonal)
                    temp_diagonal = tf.ones_like(diagonal)

                    for k in range(self.p):
                        temp_diagonal = tf.multiply(temp_diagonal, diagonal)
                        if not self.geometric:
                            temp_diagonal = tf.divide(temp_diagonal, factorial(k+1))
                        power_series = tf.add(power_series, temp_diagonal)

                    power_series = tf.linalg.diag(power_series)
                else:
                    if self.geometric:
                        power_series = tf.linalg.diag(1 / (1 - diagonal))
                    else:
                        power_series = tf.linalg.diag(tf.exp(diagonal))

                matrix_entry = tf.linalg.matmul(
                    flanking_factor,
                    tf.linalg.matmul(
                        power_series,
                        tf.transpose(flanking_factor, perm=[1, 0])
                    )
                )

                flattened_k_matrix = flattened_k_matrix.write(matrix_idx, matrix_entry)
                matrix_idx += 1

        k_matrix = tf.reshape(flattened_k_matrix.stack(), (len(X), len(X2)))

        if self.normalize:
            normalized_k_matrix = normalize(k_matrix)
            return normalized_k_matrix

        return k_matrix

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X.
        :param X: array of N graph objects (represented as adjacency matrices of varying sizes).
        :return: N x 1 array.
        """
        return tf.linalg.tensor_diag_part(self.K(X))
