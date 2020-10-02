"""
Script for training a model to predict properties using an Attentive Neural Process.
"""

import argparse

from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import transform_data, TaskDataLoader, featurise_mols
from Attentive_NP.attentive_np import AttentiveNP


def main(task, path, representation, use_pca, n_trials, test_set_size, batch_size, lr, iterations, r_size, det_encoder_hidden_size,
         det_encoder_n_hidden, lat_encoder_hidden_size, lat_encoder_n_hidden, decoder_hidden_size, decoder_n_hidden):
    """
    :param task: str specifying the task name. One of [Photoswitch, ESOL, FreeSolv, Lipophilicity].
    :param path: str specifying the path to the dataset csv file
    :param representation: str specifying the representation. One of [fingerprints, fragments, fragprints]
    :param use_pca: bool specifying whether or not to use PCA to perform Principal Components Regression
    :param n_trials: int specifying the number of random train/test splits.
    :param test_set_size: float specifying the train/test split ratio. e.g. 0.2 is 80/20 train/test split
    :param batch_size: int specifying the number of samples to take of the context set, given the number of
    context points that should be selected.
    :param lr: float specifying the learning rate.
    :param iterations: int specifying the number of training iterations
    :param r_size: Dimensionality of context encoding r.
    :param det_encoder_hidden_size: Dimensionality of deterministic encoder hidden layers.
    :param det_encoder_n_hidden: Number of deterministic encoder hidden layers.
    :param lat_encoder_hidden_size: Dimensionality of latent encoder hidden layers.
    :param lat_encoder_n_hidden: Number of latent encoder hidden layers.
    :param decoder_hidden_size: Dimensionality of decoder hidden layers.
    :param decoder_n_hidden: Number of decoder hidden layers.
    :return:
    """

    data_loader = TaskDataLoader(task, path)
    smiles_list, y = data_loader.load_property_data()
    X = featurise_mols(smiles_list, representation)
    y_size = 1

    # If True we perform Principal Components Regression

    if use_pca:
        n_components = 50
    else:
        n_components = None

    r2_list = []
    rmse_list = []
    mae_list = []

    # We pre-allocate arrays for plotting confidence-error curves

    _, _, _, y_test = train_test_split(X, y, test_size=test_set_size)  # To get test set size

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
    j = 0  # index for saving results

    for i in range(0, n_trials):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)

        if task != 'Photoswitch':
            # Artificially create a 80/10/10 train/validation/test split discarding the validation set.
            split_in_two = int(len(y_test) / 2)
            X_test = X_test[0:split_in_two]
            y_test = y_test[0:split_in_two]

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        #  We standardise the outputs but leave the inputs unchanged

        X_train, y_train, X_test, _, y_scaler = transform_data(X_train, y_train, X_test, y_test,
                                                               n_components=n_components, use_pca=use_pca)

        X_train = torch.from_numpy(X_train).float().unsqueeze(dim=0)
        X_test = torch.from_numpy(X_test).float().unsqueeze(dim=0)
        y_train = torch.from_numpy(y_train).float().unsqueeze(dim=0)

        m = AttentiveNP(x_size=X_train.shape[2], y_size=y_size, r_size=r_size,
                        det_encoder_hidden_size=8,
                        det_encoder_n_hidden=det_encoder_n_hidden,
                        lat_encoder_hidden_size=16,
                        lat_encoder_n_hidden=lat_encoder_n_hidden,
                        decoder_hidden_size=8,
                        decoder_n_hidden=decoder_n_hidden,
                        lr=0.001, attention_type="multihead")

        print('...training.')

        np.random.seed(42)

        m.train(X_train, y_train, batch_size=32, iterations=500, print_freq=None)

        # Now, the context set comprises the training x / y values, the target set comprises the test x values.

        y_pred, y_var = m.predict(X_train, y_train, X_test, n_samples=100)

        # For producing the calibration curve

        for k in [0.13, 0.26, 0.39, 0.53, 0.68, 0.85, 1.04, 1.15, 1.28, 1.44, 1.645, 1.96]:
            a = (y_test < y_scaler.inverse_transform(y_pred + k * np.sqrt(y_var)))
            b = (y_test > y_scaler.inverse_transform(y_pred - k * np.sqrt(y_var)))
            prediction_prop[i].append(np.argwhere((a == True) & (b == True)).shape[0] / len(y_test))

        y_pred = y_scaler.inverse_transform(y_pred)

        # Compute scores for confidence curve plotting.

        ranked_confidence_list = np.argsort(y_var.numpy(), axis=0).flatten()

        for k in range(len(y_test)):
            # Construct the RMSE error for each level of confidence

            conf = ranked_confidence_list[0:k + 1]
            rmse = np.sqrt(mean_squared_error(y_test[conf], y_pred[conf]))
            rmse_confidence_list[i, k] = rmse

            # Construct the MAE error for each level of confidence

            mae = mean_absolute_error(y_test[conf], y_pred[conf])
            mae_confidence_list[i, k] = mae

        # Output Standardised RMSE and RMSE on Train Set

        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

        j += 1

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)))

    # Plot confidence-error curves

    # 1e-14 instead of 0 to stop weirdness with len(y_test) = 29
    confidence_percentiles = np.arange(1e-14, 100, 100 / len(y_test))

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
    plt.ylabel('RMSE (nm)')
    plt.ylim([0, np.max(upper) + 1])
    plt.xlim([0, 100 * ((len(y_test) - 1) / len(y_test))])
    plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
    plt.savefig(task + '/results/ANP/{}_{}_confidence_curve_rmse.png'.format(representation, task))
    plt.close()

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
    plt.ylabel('MAE (nm)')
    plt.ylim([0, np.max(upper) + 1])
    plt.xlim([0, 100 * ((len(y_test) - 1) / len(y_test))])
    plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
    plt.savefig(task + '/results/ANP/{}_{}_confidence_curve_mae.png'.format(representation, task))
    plt.close()

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
    plt.savefig(task + '/results/ANP/{}_{}_calibration_curve.png'.format(representation, task))
    plt.show()

    np.savetxt(task + '/results/ANP/{}_{}_mean_props'.format(representation, task), mean_props)
    np.savetxt(task + '/results/ANP/{}_{}_sd_props'.format(representation, task), sd_props)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task', default='FreeSolv',
                        help='Task name. One of [Photoswitch, ESOL, FreeSolv, Lipophilicity].')
    parser.add_argument('-p', '--path', default='../datasets/FreeSolv.csv',
                        help='Path to task dataset csv file.')
    parser.add_argument('-r', '--representation', default='fragments',
                        help='Descriptor type. One of [fingerprints, fragments, fragprints.')
    parser.add_argument('-pca', '--use_pca', type=bool, default=False,
                        help='If true, apply PCA to data (50 components).')
    parser.add_argument('-n', '--n_trials', type=int, default=20,
                        help='Number of train test splits to try.')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='Fraction of Dataset to use as test set.')
    parser.add_argument('-b', '--batch_size', type=int, default=10,
                        help='The number of samples to take of the context set, given the number of'
                             ' context points that should be selected.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='The learning rate.')
    parser.add_argument('-i', '--iterations', type=int, default=500,
                        help='Number of training iterations.')
    parser.add_argument('-rs', '--r_size', type=int, default=8,
                        help='Dimensionality of context encoding, r.')
    parser.add_argument('-dhs', '--det_encoder_hidden_size', type=int, default=32,
                        help='Dimensionality of deterministic encoder hidden layers.')
    parser.add_argument('-dnh', '--det_encoder_n_hidden', type=int, default=2,
                        help='Number of deterministic encoder hidden layers.')
    parser.add_argument('-lhs', '--lat_encoder_hidden_size', type=int, default=32,
                        help='Dimensionality of latent encoder hidden layers.')
    parser.add_argument('-lnh', '--lat_encoder_n_hidden', type=int, default=2,
                        help='Number of latent encoder hidden layers.')
    parser.add_argument('-dhss', '--decoder_hidden_size', type=int, default=32,
                        help='Dimensionality of decoder hidden layers.')
    parser.add_argument('-dnhn', '--decoder_n_hidden', type=int, default=2,
                        help='Number of decoder hidden layers.')

    args = parser.parse_args()

    main(args.task, args.path, args.representation, args.use_pca, args.n_trials, args.test_set_size, args.batch_size,
         args.learning_rate, args.iterations, args.r_size, args.det_encoder_hidden_size, args.det_encoder_n_hidden,
         args.lat_encoder_hidden_size, args.lat_encoder_n_hidden, args.decoder_hidden_size, args.decoder_n_hidden)
