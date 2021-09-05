# Author: Henry Moss & Ryan-Rhys Griffiths
"""
Molecule kernels for Gaussian Process Regression implemented in GPflow.
"""

import gpflow
import tensorflow as tf


class RandomWalk(gpflow.kernels.Kernel):
    def __init__(self, uniform_probabilities=False, geometric=True, weight=0.1):
        super().__init__()
        self.uniform_probabilities=uniform_probabilities
        self.geometric = geometric
        self.weight = weight

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        X_is_X2 = X == X2

        eigenvals, eigenvecs = tf.linalg.eigh(X)
        flanking_factors = self._generate_flanking_factors(eigenvecs)

        if X_is_X2:
            eigenvals_2, eigenvecs_2 = eigenvals, eigenvecs
            flanking_factors_2 = flanking_factors
        else:
            eigenvals_2, eigenvecs_2 = tf.linalg.eigh(X2)
            flanking_factors_2 = self._generate_flanking_factors(eigenvecs_2)

        k_matrix = tf.zeros((X.shape[0], X2.shape[0]))

        for i in k_matrix.shape[0]:
            for j in k_matrix.shape[1]:

                if X_is_X2 and j < i:
                    k_matrix[i, j] = k_matrix[j, i]
                    continue

                flanking_factor = tf.linalg.LinearOperatorKronecker(flanking_factors[i], flanking_factors_2[j])
                diagonal = self.weight * tf.linalg.LinearOperatorKronecker(eigenvecs[i], eigenvecs_2[j])

                if self.geometric:
                    power_series = tf.linalg.diag(1 / 1 - diagonal)
                else:
                    power_series = tf.linalg.diag(tf.exp(diagonal))

                k_matrix[i, j] = tf.linalg.matmul(
                    flanking_factor,
                    tf.linalg.matmul(
                        power_series,
                        tf.transpose(flanking_factor, perm=[-2, -1])
                    )
                )

        return k_matrix

    def K_diag(self, X):
        return tf.linalg.tensor_diag_part(self.K(X))

    def _generate_flanking_factors(self, eigenvecs):
        flanking_factors = []

        for eigenvec in eigenvecs:
            start_stop_probs = tf.ones((1, eigenvec.shape[0]))
            if self.uniform_probabilities:
                start_stop_probs = tf.divide(start_stop_probs, eigenvec.shape(0))

            flanking_factors.append(
                tf.linalg.matmul(start_stop_probs, eigenvec)
            )

        return flanking_factors
