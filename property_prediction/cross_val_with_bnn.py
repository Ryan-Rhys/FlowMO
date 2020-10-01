# Author: Ryan-Rhys Griffiths
"""
Script for cross-validation for the Black box alpha-divergence minimisation Bayesian neural network.
"""

import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from BNN.bb_alpha import BB_alpha
from BNN.bnn_utils import load_reg_data
from data_utils import transform_data, TaskDataLoader, featurise_mols


def main(path, task, representation, use_pca, test_set_size):
    """
    :param path: str specifying path to dataset.
    :param task: str specifying the task. One of ['Photoswitch', 'ESOL', 'FreeSolv', 'Lipophilicity']
    :param representation: str specifying the molecular representation. One of [fingerprints, 'fragments', 'fragprints']
    :param use_pca: bool. If True apply PCA to perform Principal Components Regression.
    :param test_set_size: float in range [0, 1] specifying fraction of dataset to use as test set
    """

    if representation == 'SMILES':
        raise Exception('SMILES is not a valid representation for the BNN')

    data_loader = TaskDataLoader(task, path)
    smiles_list, y = data_loader.load_property_data()
    X = featurise_mols(smiles_list, representation)

    # If True we perform Principal Components Regression

    if use_pca:
        n_components = 100
    else:
        n_components = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=42)

    if task != 'Photoswitch':
        # Artificially create a 80/10/10 train/validation/test split discarding the validation set.
        split_in_two = int(len(y_test) / 2)
        X_test = X_test[0:split_in_two]
        y_test = y_test[0:split_in_two]

    else:
        # We subdivide the train set in order to run cross-validation.
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    #  We standardise the outputs but leave the inputs unchanged

    _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test, n_components=n_components, use_pca=use_pca)

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    datasets, n, d, mean_y_train, std_y_train = load_reg_data(X_train, y_train, X_test, y_test)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    layer_sizes = [10, 20]
    learning_rates = [0.01, 0.001]
    batch_sizes = [16, 32, 64]
    iters = [20, 50, 100]

    best_rmse = 10000000  # a large number
    best_params = {'layer_size': 0, 'lr': 0, 'batch_size': 0, 'iterations': 0}

    for layer_size in layer_sizes:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for iteration in iters:

                    N_train = train_set_x.get_value(borrow=True).shape[0]
                    N_test = test_set_x.get_value(borrow=True).shape[0]
                    layer_sizes = [d, layer_size, layer_size, len(mean_y_train)]
                    n_samples = 100
                    alpha = 0.5
                    learning_rate = lr
                    v_prior = 1.0
                    batch_size = batch_size
                    print('... building model')
                    sys.stdout.flush()
                    bb_alpha = BB_alpha(layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size,
                                        train_set_x, train_set_y, N_train, test_set_x, test_set_y, N_test, mean_y_train,
                                        std_y_train)
                    print('... training')
                    sys.stdout.flush()

                    test_error, test_ll = bb_alpha.train_ADAM(iteration)

                    print('Test RMSE: ', test_error)
                    print('Test ll: ', test_ll)

                    samples = bb_alpha.sample_predictive_distribution(X_test)
                    y_pred = np.mean(samples, axis=0)

                    # Output Standardised RMSE and RMSE on Train Set

                    train_samples = bb_alpha.sample_predictive_distribution(X_train)
                    y_pred_train = np.mean(train_samples, axis=0)

                    train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    train_rmse = np.sqrt(
                        mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(y_pred_train)))
                    print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
                    print("Train RMSE: {:.3f}".format(train_rmse))

                    score = r2_score(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred))
                    rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred)))
                    mae = mean_absolute_error(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(y_pred))

                    print("\nR^2: {:.3f}".format(score))
                    print("RMSE: {:.3f}".format(rmse))
                    print("MAE: {:.3f}".format(mae))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params['lr'] = lr
                        best_params['batch_size'] = batch_size
                        best_params['iterations'] = iteration
                        best_params['layer_size'] = layer_size
                    print('Best parameters are \n')
                    print(best_params)

    print('Final best parameters are \n')
    print(best_params)

    with open(f'cross_val_hypers/{task}/BNN/hypers_{representation}.txt', 'w') as f:
        f.write(str(best_params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../datasets/FreeSolv.csv',
                        help='Path to the csv file for the task.')
    parser.add_argument('-t', '--task', type=str, default='FreeSolv',
                        help='str specifying the task. One of [Photoswitch, ESOL, FreeSolv, Lipophilicity].')
    parser.add_argument('-r', '--representation', type=str, default='fingerprints',
                        help='str specifying the molecular representation. '
                             'One of [fingerprints, fragments, fragprints].')
    parser.add_argument('-pca', '--use_pca', type=bool, default=False,
                        help='If True apply PCA to perform Principal Components Regression.')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='float in range [0, 1] specifying fraction of dataset to use as test set')

    args = parser.parse_args()

    main(args.path, args.task, args.representation, args.use_pca, args.test_set_size)
