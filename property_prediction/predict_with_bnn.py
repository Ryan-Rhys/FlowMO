# Author: Ryan-Rhys Griffiths
"""
Script for training a model to predict properties using a Black Box alpha-divergence
minimisation Bayesian neural network.
"""

import argparse
import sys

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from BNN.bb_alpha import BB_alpha
from BNN.bnn_utils import load_reg_data
from data_utils import transform_data, TaskDataLoader, featurise_mols


def main(path, task, representation, use_pca, n_trials, test_set_size, use_rmse_conf, precompute_repr):
    """
    :param path: str specifying path to dataset.
    :param task: str specifying the task. One of ['Photoswitch', 'ESOL', 'FreeSolv', 'Lipophilicity']
    :param representation: str specifying the molecular representation. One of ['SMILES, fingerprints, 'fragments', 'fragprints']
    :param use_pca: bool. If True apply PCA to perform Principal Components Regression.
    :param n_trials: int specifying number of random train/test splits to use
    :param test_set_size: float in range [0, 1] specifying fraction of dataset to use as test set
    :param use_rmse_conf: bool specifying whether to compute the rmse confidence-error curves or the mae confidence-
    error curves. True is the option for rmse.
    :param precompute_repr: bool indicating whether to precompute representations or not.
    """

    data_loader = TaskDataLoader(task, path)
    smiles_list, y = data_loader.load_property_data()
    X = featurise_mols(smiles_list, representation)

    if precompute_repr:
        if representation == 'SMILES':
            with open(f'precomputed_representations/{task}_{representation}.txt', 'w') as f:
                for smiles in X:
                    f.write(smiles + '\n')
        else:
            np.savetxt(f'precomputed_representations/{task}_{representation}.txt', X)

    # If True we perform Principal Components Regression

    if use_pca:
        n_components = 100
    else:
        n_components = None

    r2_list = []
    rmse_list = []
    mae_list = []

    # We pre-allocate arrays for plotting confidence-error curves

    _, _, _, y_test = train_test_split(X, y, test_size=test_set_size, random_state=42)  # To get test set size

    # Photoswitch dataset requires 80/20 splitting. Other datasets are 80/10/10.

    if task != 'Photoswitch':
        split_in_two = int(len(y_test)/2)
        n_test = split_in_two
    else:
        n_test = len(y_test)

    rmse_confidence_list = np.zeros((n_trials, n_test))
    mae_confidence_list = np.zeros((n_trials, n_test))

    # For Calibration curve

    prediction_prop = [[] for _ in range(n_trials)]

    print('\nBeginning training loop...')

    for i in range(0, n_trials):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)

        if representation == 'SMILES':

            np.savetxt(f'fixed_train_test_splits/{task}/X_train_split_{i}.txt', X_train, fmt="%s")
            np.savetxt(f'fixed_train_test_splits/{task}/X_test_split_{i}.txt', X_test, fmt="%s")
            np.savetxt(f'fixed_train_test_splits/{task}/y_train_split_{i}.txt', y_train)
            np.savetxt(f'fixed_train_test_splits/{task}/y_test_split_{i}.txt', y_test)

        else:

            if task != 'Photoswitch':

                # Artificially create a 80/10/10 train/validation/test split discarding the validation set.
                split_in_two = int(len(y_test)/2)
                X_test = X_test[0:split_in_two]
                y_test = y_test[0:split_in_two]

            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

            #  We standardise the outputs but leave the inputs unchanged

            _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test, n_components=n_components, use_pca=use_pca)

            X_train = X_train.astype(np.float64)
            X_test = X_test.astype(np.float64)

            np.random.seed(42)

            datasets, n, d, mean_y_train, std_y_train = load_reg_data(X_train, y_train, X_test, y_test)

            train_set_x, train_set_y = datasets[0]
            test_set_x, test_set_y = datasets[1]

            N_train = train_set_x.get_value(borrow=True).shape[0]
            N_test = test_set_x.get_value(borrow=True).shape[0]
            layer_sizes = [d, 20, 20, len(mean_y_train)]
            n_samples = 100
            alpha = 0.5
            learning_rate = 0.01
            v_prior = 1.0
            batch_size = 64
            print('... building model')
            sys.stdout.flush()
            bb_alpha = BB_alpha(layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size,
                                train_set_x, train_set_y, N_train, test_set_x, test_set_y, N_test, mean_y_train, std_y_train)
            print('... training')
            sys.stdout.flush()

            test_error, test_ll = bb_alpha.train_ADAM(50)

            print('Test RMSE: ', test_error)
            print('Test ll: ', test_ll)

            samples = bb_alpha.sample_predictive_distribution(X_test)
            y_pred = np.mean(samples, axis=0)
            var = np.var(samples, axis=0)

            # For producing the calibration curve

            for k in [0.13, 0.26, 0.39, 0.53, 0.68, 0.85, 1.04, 1.15, 1.28, 1.44, 1.645, 1.96]:
                a = (y_scaler.inverse_transform(y_test) < y_scaler.inverse_transform(y_pred + k * np.sqrt(var)))
                b = (y_scaler.inverse_transform(y_test) > y_scaler.inverse_transform(y_pred - k * np.sqrt(var)))
                prediction_prop[i].append(np.argwhere((a == True) & (b == True)).shape[0] / len(y_test))

            # We transform the standardised predictions back to the original data space

            y_pred = y_scaler.inverse_transform(y_pred)
            y_test = y_scaler.inverse_transform(y_test)

            # Compute scores for confidence curve plotting.

            ranked_confidence_list = np.argsort(var, axis=0).flatten()

            for k in range(len(y_test)):

                # Construct the RMSE error for each level of confidence

                conf = ranked_confidence_list[0:k+1]
                rmse = np.sqrt(mean_squared_error(y_test[conf], y_pred[conf]))
                rmse_confidence_list[i, k] = rmse

                # Construct the MAE error for each level of confidence

                mae = mean_absolute_error(y_test[conf], y_pred[conf])
                mae_confidence_list[i, k] = mae

            # Output Standardised RMSE and RMSE on Train Set

            train_samples = bb_alpha.sample_predictive_distribution(X_train)
            y_pred_train = np.mean(train_samples, axis=0)

            train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_train), y_scaler.inverse_transform(y_pred_train)))
            print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
            print("Train RMSE: {:.3f}".format(train_rmse))

            score = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            print("\nR^2: {:.3f}".format(score))
            print("RMSE: {:.3f}".format(rmse))
            print("MAE: {:.3f}".format(mae))

            r2_list.append(score)
            rmse_list.append(rmse)
            mae_list.append(mae)

    if representation != 'SMILES':

        r2_list = np.array(r2_list)
        rmse_list = np.array(rmse_list)
        mae_list = np.array(mae_list)

        print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)))
        print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)))
        print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)))

        # Plot confidence-error curves

        confidence_percentiles = np.arange(1e-14, 100, 100/len(y_test))  # 1e-14 instead of 0 to stop weirdness with len(y_test) = 29

        if use_rmse_conf:

            rmse_mean = np.mean(rmse_confidence_list, axis=0)
            rmse_std = np.std(rmse_confidence_list, axis=0)

            # We flip because we want the most confident predictions on the right-hand side of the plot

            rmse_mean = np.flip(rmse_mean)
            rmse_std = np.flip(rmse_std)

            # One-sigma error bars

            lower = rmse_mean - rmse_std
            upper = rmse_mean + rmse_std

            plt.plot(confidence_percentiles, rmse_mean, label='mean')
            plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
            plt.xlabel('Confidence Percentile')
            plt.ylabel('RMSE')
            plt.ylim([0, np.max(upper) + 1])
            plt.xlim([0, 100*((len(y_test) - 1) / len(y_test))])
            plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
            plt.savefig(task + '/results/BNN/{}_{}_confidence_curve_rmse.png'.format(representation, task))
            plt.show()

        else:

            # We plot the Mean-absolute error confidence-error curves

            mae_mean = np.mean(mae_confidence_list, axis=0)
            mae_std = np.std(mae_confidence_list, axis=0)

            mae_mean = np.flip(mae_mean)
            mae_std = np.flip(mae_std)

            lower = mae_mean - mae_std
            upper = mae_mean + mae_std

            plt.plot(confidence_percentiles, mae_mean, label='mean')
            plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
            plt.xlabel('Confidence Percentile')
            plt.ylabel('MAE')
            plt.ylim([0, np.max(upper) + 1])
            plt.xlim([0, 100 * ((len(y_test) - 1) / len(y_test))])
            plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
            plt.savefig(task + '/results/BNN/{}_{}_confidence_curve_mae.png'.format(representation, task))
            plt.show()

        # Plot the calibration curve

        mean_props = np.mean(prediction_prop, axis=0)
        sd_props = np.std(prediction_prop, axis=0)
        lower = mean_props - sd_props
        upper = mean_props + sd_props
        qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        plt.plot(qs, mean_props, label='mean')
        plt.fill_between(qs, lower, upper, alpha=0.2)
        plt.plot(qs, qs, color="red")
        plt.xlabel('q')
        plt.ylabel('C(q)')
        plt.savefig(task + '/results/BNN/{}_{}_calibration_curve.png'.format(representation, task))
        plt.show()

        np.savetxt(task + '/results/BNN/{}_{}_mean_props'.format(representation, task), mean_props)
        np.savetxt(task + '/results/BNN/{}_{}_sd_props'.format(representation, task), sd_props)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../datasets/FreeSolv.csv',
                        help='Path to the csv file for the task.')
    parser.add_argument('-t', '--task', type=str, default='FreeSolv',
                        help='str specifying the task. One of [Photoswitch, ESOL, FreeSolv, Lipophilicity].')
    parser.add_argument('-r', '--representation', type=str, default='fragments',
                        help='str specifying the molecular representation. '
                             'One of [SMILES, fingerprints, fragments, fragprints].')
    parser.add_argument('-pca', '--use_pca', type=bool, default=False,
                        help='If True apply PCA to perform Principal Components Regression.')
    parser.add_argument('-n', '--n_trials', type=int, default=5,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='float in range [0, 1] specifying fraction of dataset to use as test set')
    parser.add_argument('-rms', '--use_rmse_conf', type=bool, default=True,
                        help='bool specifying whether to compute the rmse confidence-error curves or the mae '
                             'confidence-error curves. True is the option for rmse.')
    parser.add_argument('-pr', '--precompute_repr', type=bool, default=True,
                        help='bool indicating whether to precompute representations')

    args = parser.parse_args()

    main(args.path, args.task, args.representation, args.use_pca, args.n_trials, args.test_set_size, args.use_rmse_conf,
         args.precompute_repr)
