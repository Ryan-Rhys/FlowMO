"""
Module containing the Bayesian Neural Network.
"""

import time

import numpy as np
import scipy.optimize as spo
import theano
import theano.tensor as T

import network
from bnn_utils import LogSumExp, casting, adam, global_optimization


class BB_alpha:

    def __init__(self, layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size, X_train, y_train, N_train, \
                 X_val, y_val, N_val, mean_y_train, std_y_train):

        # 29 October 17:55 - y_min and y_max added as arguments on 23rd October. Instance variables or just supply to the function?
        # 11 November 08:12 - They will need to be instance variables of the class.

        layer_sizes[0] = layer_sizes[0] + 1
        self.batch_size = batch_size
        self.N_train = N_train
        self.X_train = X_train
        self.y_train = y_train

        # 11 November 08:17, removing y_min and y_max as parameters to the constructor above and creating them as instance variables here.

        self.y_min = np.min(y_train.eval())
        self.y_max = np.max(y_train.eval())

        self.N_val = N_val
        self.X_val = X_val
        self.y_val = y_val

        self.mean_y_train = mean_y_train
        self.std_y_train = std_y_train
        self.n_samples = n_samples

        # We create the network

        self.network = network.Network(layer_sizes, n_samples, v_prior, N_train)

        # index to a batch

        index = T.lscalar()

        # We create the input and output variables. The input will be a minibatch replicated n_samples times

        self.x = T.matrix('x')
        self.y = T.matrix('y', dtype='float32')

        # The logarithm of the values for the likelihood factors

        ll_train = self.network.log_likelihood_values(self.x, self.y, 0.0, 1.0)
        ll_val = self.network.log_likelihood_values(self.x, self.y, mean_y_train, std_y_train)

        # The energy function for black-box alpha

        self.estimate_marginal_ll = -1.0 * N_train / (self.x.shape[0] * alpha) * \
                                    T.sum(LogSumExp(alpha * (T.sum(ll_train, 2) - self.network.log_f_hat()), 0) + \
                                          T.log(1.0 / n_samples)) - self.network.log_normalizer_q() + \
                                    self.network.log_Z_prior()

        # We create a theano function for updating q

        self.process_minibatch = theano.function([index], self.estimate_marginal_ll, \
                                                 updates=adam(self.estimate_marginal_ll, self.network.params,
                                                              learning_rate), \
                                                 givens={
                                                     self.x: self.X_train[index * batch_size: (index + 1) * batch_size],
                                                     self.y: self.y_train[
                                                             index * batch_size: (index + 1) * batch_size]})

        # We create a theano function for making predictions

        self.error_minibatch_train = theano.function([index],
                                                     T.sum((T.mean(self.network.output(self.x), 0, keepdims=True)[0, :,
                                                            :] - self.y) ** 2) / layer_sizes[-1],
                                                     givens={self.x: self.X_train[
                                                                     index * batch_size: (index + 1) * batch_size],
                                                             self.y: self.y_train[
                                                                     index * batch_size: (index + 1) * batch_size]})

        self.error_minibatch_val = theano.function([index],
                                                   T.sum((T.mean(self.network.output(self.x), 0, keepdims=True)[0, :,
                                                          :] * std_y_train + mean_y_train - self.y) ** 2) / layer_sizes[
                                                       -1],
                                                   givens={
                                                       self.x: self.X_val[index * batch_size: (index + 1) * batch_size],
                                                       self.y: self.y_val[
                                                               index * batch_size: (index + 1) * batch_size]})

        self.ll_minibatch_val = theano.function([index], T.sum(LogSumExp(T.sum(ll_val, 2), 0) + T.log(1.0 / n_samples)), \
                                                givens={
                                                    self.x: self.X_val[index * batch_size: (index + 1) * batch_size],
                                                    self.y: self.y_val[index * batch_size: (index + 1) * batch_size]})

        self.ll_minibatch_train = theano.function([index],
                                                  T.sum(LogSumExp(T.sum(ll_train, 2), 0) + T.log(1.0 / n_samples)), \
                                                  givens={self.x: self.X_train[
                                                                  index * batch_size: (index + 1) * batch_size],
                                                          self.y: self.y_train[
                                                                  index * batch_size: (index + 1) * batch_size]})

        self.target_lbfgs_grad = theano.function([], T.grad(self.estimate_marginal_ll, self.network.params),
                                                 givens={self.x: self.X_train, self.y: self.y_train})

        self.target_lbfgs_objective = theano.function([], self.estimate_marginal_ll,
                                                      givens={self.x: self.X_train, self.y: self.y_train})

        # 26 November 16:51 - Will subbing prediction in directly change anything?

        prediction = self.network.output(self.x) * std_y_train[None, None, :] + mean_y_train[None, None, :]
        self.predict = theano.function([self.x], prediction)
        self.function_grid = theano.function([self.x], prediction[0, :, 0])
        self.function_scalar = theano.function([self.x], prediction[0, 0, 0])
        self.function_scalar_gradient = theano.function([self.x], T.grad(prediction[0, 0, 0], self.x))

        self.network.update_randomness()

    def sample_predictive_distribution(self, x):

        noise_std = np.tile(np.expand_dims(self.std_y_train * \
                                           np.sqrt(np.exp(self.network.log_v_noise.get_value())), 0),
                            [self.n_samples, x.shape[0], 1])
        y = self.predict(x)
        return y + noise_std * np.random.randn(self.n_samples, x.shape[0], len(self.std_y_train))

    def train_LBFGS(self, n_epochs):

        initial_params = theano.function([], self.network.params)()

        params_shapes = [s.shape for s in initial_params]

        def de_vectorize_params(params):
            ret = []
            for shape in params_shapes:
                if len(shape) == 2 or len(shape) == 3:
                    ret.append(params[: np.prod(shape)].reshape(shape))
                    params = params[np.prod(shape):]
                elif len(shape) == 1:
                    ret.append(params[: np.prod(shape)])
                    params = params[np.prod(shape):]
                else:
                    ret.append(params[0])
                    params = params[1:]
            return ret

        def vectorize_params(params):
            return np.concatenate([np.array(s).flatten() for s in params])

        def set_params(params):
            for i in range(len(params)):
                self.network.params[i].set_value(params[i])

        def objective(params):
            params = np.array(params).astype(theano.config.floatX)
            params = de_vectorize_params(params)
            set_params(params)
            obj = np.array(self.target_lbfgs_objective(), dtype=np.float64)
            grad = np.array(vectorize_params(self.target_lbfgs_grad()), dtype=np.float64)
            return obj, grad

        initial_params = vectorize_params(initial_params)
        x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(objective, initial_params, bounds=None, iprint=1, maxiter=n_epochs)
        set_params(de_vectorize_params(np.array(x_opt).astype(theano.config.floatX)))

        n_batches_train = np.int(np.ceil(1.0 * self.N_train / self.batch_size))
        n_batches_val = np.int(np.ceil(1.0 * self.N_val / self.batch_size))

        error_train = 0
        ll_train = 0
        for idxs in range(n_batches_train):
            error_train += self.error_minibatch_train(idxs)
            ll_train += self.ll_minibatch_train(idxs)
        error_train /= self.N_train
        error_train = np.sqrt(error_train)
        ll_train /= self.N_train

        error_val = 0
        ll_val = 0
        for idxs in range(n_batches_val):
            error_val += self.error_minibatch_val(idxs)
            ll_val += self.ll_minibatch_val(idxs)
        error_val /= self.N_val
        error_val = np.sqrt(error_val)
        ll_val /= self.N_val

        return error_val, ll_val

    def batched_thompson_sampling(self, bb_alpha_con, q, lower, upper, bb_alpha_samples):

        '''
        q = number of samples
        lower = lowest x value in training data
        upper = highest x value in training data (specifying range?)
        '''

        grid_size = 10000
        grid = casting(lower + np.random.rand(grid_size, len(lower)) * (upper - lower))

        def sigmoid(x): return 1.0 / (1.0 + T.exp(-x))

        x = T.matrix('x', dtype=theano.config.floatX)

        prediction_probs = T.exp(LogSumExp(bb_alpha_con.network.output(self.x), 0) + T.log(
            1.0 / bb_alpha_samples)) ** 30  # 2-D array of size (n_samples, 2) where column 1 gives the probability of the constraint being unsatisfied and column two gives the probabilty of the constraint being satisfied.

        # 19 November 20:45 prediction_lg is the logistic function applied to the NN output.

        prediction_lg = sigmoid(4.0 * (self.network.output(self.x) - self.y_max) / (self.y_min - self.y_max) - 2.0)
        predict_lg = theano.function([self.x], prediction_lg)
        function_grid_lg = theano.function([self.x], -prediction_lg[0, :, 0] * T.reshape(prediction_probs[:, :, 1],
                                                                                         [T.shape(self.x)[0], 1])[:, 0])
        function_scalar_lg = theano.function([self.x], -prediction_lg[0, 0, 0] *
                                             T.reshape(prediction_probs[:, :, 1], [T.shape(self.x)[0], 1])[0, 0])
        function_scalar_gradient_lg = theano.function([self.x], T.grad(
            -prediction_lg[0, 0, 0] * T.reshape(prediction_probs[:, :, 1], [T.shape(self.x)[0], 1])[0, 0], self.x))

        self.network.update_randomness(grid_size)
        X_numpy = \
        global_optimization(grid, lower, upper, function_grid_lg, function_scalar_lg, function_scalar_gradient_lg)[0]
        for i in range(1, q):
            self.network.update_randomness(grid_size)
            new_point = \
            global_optimization(grid, lower, upper, function_grid_lg, function_scalar_lg, function_scalar_gradient_lg)[
                0]  # new_point.shape = (1,2)
            X_numpy = casting(np.concatenate([X_numpy, new_point], 0))
            print(i, X_numpy)

        samples = self.predict(X_numpy)

        print("Predictive mean at selected points:\n", np.mean(samples, 0)[:, 0])

        return X_numpy

    def train_ADAM(self, n_epochs):

        n_batches_train = np.int(np.ceil(1.0 * self.N_train / self.batch_size))
        n_batches_val = np.int(np.ceil(1.0 * self.N_val / self.batch_size))
        for i in range(n_epochs):
            permutation = np.random.choice(range(n_batches_train), n_batches_train, replace=False)
            start = time.time()
            for idxs in range(n_batches_train):
                #                if idxs % 10 == 9:
                #                    self.network.update_randomness()
                self.network.update_randomness()
                ret = self.process_minibatch(permutation[idxs])
            end = time.time()

            # We evaluate the performance on the test data

            error_train = 0
            ll_train = 0
            for idxs in range(n_batches_train):
                error_train += self.error_minibatch_train(idxs)
                ll_train += self.ll_minibatch_train(idxs)
            error_train /= self.N_train
            error_train = np.sqrt(error_train)
            ll_train /= self.N_train

            error_val = 0
            ll_val = 0
            for idxs in range(n_batches_val):
                error_val += self.error_minibatch_val(idxs)
                ll_val += self.ll_minibatch_val(idxs)
            error_val /= self.N_val
            error_val = np.sqrt(error_val)
            ll_val /= self.N_val

            print(i, error_train, ll_train, error_val, ll_val, end - start)

        error_train = 0
        ll_train = 0
        for idxs in range(n_batches_train):
            error_train += self.error_minibatch_train(idxs)
            ll_train += self.ll_minibatch_train(idxs)
        error_train /= self.N_train
        error_train = np.sqrt(error_train)
        ll_train /= self.N_train

        error_val = 0
        ll_val = 0
        for idxs in range(n_batches_val):
            error_val += self.error_minibatch_val(idxs)
            ll_val += self.ll_minibatch_val(idxs)
        error_val /= self.N_val
        error_val = np.sqrt(error_val)
        ll_val /= self.N_val

        return error_val, ll_val
