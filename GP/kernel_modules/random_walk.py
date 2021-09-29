# Author: Henry Moss & Ryan-Rhys Griffiths
"""
Molecule kernels for Gaussian Process Regression implemented in GPflow.
"""

import gpflow
import numpy as np
import tensorflow as tf

from .kernel_utils import normalize


class RandomWalk(gpflow.kernels.Kernel):
    def __init__(self, uniform_probabilities=False, series_type='geometric', weight=0.1, normalize=True):
        super().__init__()
        self.uniform_probabilities=uniform_probabilities
        if series_type == 'geometric':
            self.geometric = True
        elif series_type == 'exponential':
            self.geometric = False
        self.weight = weight
        self.normalize = normalize

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        X_is_X2 = X == X2

        eigenvecs, eigenvals = [], []
        eigenvecs_2, eigenvals_2 = [], []

        for adj_mat in X:
            val, vec = tf.linalg.eigh(tf.cast(adj_mat, tf.float64))
            eigenvals.append(val)
            eigenvecs.append(vec)

        flanking_factors = self._generate_flanking_factors(eigenvecs)

        if X_is_X2:
            eigenvals_2, eigenvecs_2 = eigenvals, eigenvecs
            flanking_factors_2 = flanking_factors
        else:
            for adj_mat in X2:
                val, vec = tf.linalg.eigh(tf.cast(adj_mat, tf.float64))
                eigenvals_2.append(val)
                eigenvecs_2.append(vec)
            flanking_factors_2 = self._generate_flanking_factors(eigenvecs_2)

        k_matrix = np.zeros((len(X), len(X2)))

        for idx_1 in range(k_matrix.shape[0]):
            for idx_2 in range(k_matrix.shape[1]):

                if X_is_X2 and idx_2 < idx_1:
                    k_matrix[idx_1, idx_2] = k_matrix[idx_2, idx_1]
                    continue

                flanking_factor = tf.linalg.LinearOperatorKronecker(
                    [tf.linalg.LinearOperatorFullMatrix(flanking_factors[idx_1]),
                     tf.linalg.LinearOperatorFullMatrix(flanking_factors_2[idx_2])
                     ]).to_dense()

                diagonal = self.weight * tf.linalg.LinearOperatorKronecker(
                    [tf.linalg.LinearOperatorFullMatrix(tf.expand_dims(eigenvals[idx_1], axis=0)),
                     tf.linalg.LinearOperatorFullMatrix(tf.expand_dims(eigenvals_2[idx_2], axis=0))
                     ]).to_dense()

                if self.geometric:
                    power_series = tf.linalg.diag(1 / (1 - diagonal))
                else:
                    power_series = tf.linalg.diag(tf.exp(diagonal))

                k_matrix[idx_1, idx_2] = tf.linalg.matmul(
                    flanking_factor,
                    tf.linalg.matmul(
                        power_series,
                        tf.transpose(flanking_factor, perm=[1, 0])
                    )
                )

        if self.normalize:
            return tf.convert_to_tensor(normalize(k_matrix))

        return tf.convert_to_tensor(k_matrix)

    def K_diag(self, X):
        return tf.linalg.tensor_diag_part(self.K(X))

    def _generate_flanking_factors(self, eigenvecs):
        flanking_factors = []

        for eigenvec in eigenvecs:
            start_stop_probs = tf.ones((1, eigenvec.shape[0]), tf.float64)
            if self.uniform_probabilities:
                start_stop_probs = tf.divide(start_stop_probs, eigenvec.shape(0))

            flanking_factors.append(
                tf.linalg.matmul(start_stop_probs, eigenvec)
            )

        return flanking_factors
