"""
Utility function module for the Bayesian Neural Network.
"""

import numpy as np
import scipy.optimize as spo
import theano
import theano.tensor as T


def load_reg_data(X_tr_reg, y_tr_reg, X_te_reg, y_te_reg):

    def shared_reg_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

        return shared_x, T.cast(shared_y, 'float32')

    mean_y_tr_reg = 0 * np.mean(y_tr_reg, 0)
    std_y_tr_reg = 1.0 + 0 * np.std(y_tr_reg, 0)

    train_set = X_tr_reg, y_tr_reg
    test_set = X_te_reg, y_te_reg

    train_set_x, train_set_y = shared_reg_dataset(train_set)
    test_set_x, test_set_y = shared_reg_dataset(test_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval, train_set[ 0 ].shape[ 0 ], train_set[ 0 ].shape[ 1 ], mean_y_tr_reg, std_y_tr_reg


def casting(x):
    return np.array(x).astype(theano.config.floatX)


def global_optimization(grid, lower, upper, function_grid, function_scalar, function_scalar_gradient):
    grid_values = function_grid(grid)  # grid_values.shape = (10000,)    grid.shape = (10000, 2)
    best = grid_values.argmin()  # gives index of minimum value in grid_values

    # We solve the optimization problem

    X_initial = grid[best: (best + 1), :]

    def objective(X):
        X = casting(X)
        X = X.reshape((1, grid.shape[1]))
        value = function_scalar(X)
        gradient_value = function_scalar_gradient(X).flatten()
        return np.float(value), gradient_value.astype(np.float)

    lbfgs_bounds = zip(lower.tolist(), upper.tolist())
    x_optimal, y_opt, opt_info = spo.fmin_l_bfgs_b(objective, X_initial, bounds=lbfgs_bounds, iprint=0, maxiter=150)
    x_optimal = x_optimal.reshape((1, grid.shape[1]))

    return x_optimal, y_opt


def make_batches(N_data, batch_size):
    return [slice(i, min(i + batch_size, N_data)) for i in range(0, N_data, batch_size)]


def adam(loss, all_params, learning_rate=0.001):
    b1 = 0.9
    b2 = 0.999
    e = 1e-8
    gamma = 1 - 1e-8
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        m = b1 * m_previous + (1 - b1) * g  # (Update biased first moment estimate)
        v = b2 * v_previous + (1 - b2) * g ** 2  # (Update biased second raw moment estimate)
        m_hat = m / (1 - b1 ** t)  # (Compute bias-corrected first moment estimate)
        v_hat = v / (1 - b2 ** t)  # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e)  # (Update parameters)
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta))
    updates.append((t, t + 1.))
    return updates


def LogSumExp(x, axis=None):
    """
    Compute the LogSumExp.
    :param x: a matrix of dimension [n_samples, batch_size, n_features].
    :param axis: axis along which to compute the LogSumExp.
    :return LogSumExp
    """
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
