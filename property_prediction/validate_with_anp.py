"""
Script for performing validation for the Attentive Neural Process.
"""

import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import transform_data, TaskDataLoader, featurise_mols
from Attentive_NP.attentive_np import AttentiveNP


def main(task, path, representation, use_pca, test_set_size, r_size, det_encoder_n_hidden, lat_encoder_n_hidden,
         decoder_n_hidden):
    """
    :param task: str specifying the task name. One of [Photoswitch, ESOL, FreeSolv, Lipophilicity].
    :param path: str specifying the path to the photoswitches.csv file
    :param representation: str specifying the representation. One of [fingerprints, fragments, fragprints]
    :param use_pca: bool specifying whether or not to use PCA to perform Principal Components Regression
    :param test_set_size: float specifying the train/test split ratio. e.g. 0.2 is 80/20 train/test split
    :param r_size: Dimensionality of context encoding r.
    :param det_encoder_n_hidden: Number of deterministic encoder hidden layers.
    :param lat_encoder_n_hidden: Number of latent encoder hidden layers.
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

    print('\nBeginning training loop...')
    j = 0  # index for saving results

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

    X_train, y_train, X_test, _, y_scaler = transform_data(X_train, y_train, X_test, y_test,
                                                           n_components=n_components, use_pca=use_pca)

    X_train = torch.from_numpy(X_train).float().unsqueeze(dim=0)
    X_test = torch.from_numpy(X_test).float().unsqueeze(dim=0)
    y_train = torch.from_numpy(y_train).float().unsqueeze(dim=0)

    det_encoder_hidden_sizes = [8, 16]
    lat_encoder_hidden_sizes = [8, 16]
    decoder_hidden_sizes = [8, 16]
    learning_rates = [0.01, 0.001]
    batch_sizes = [16, 32]
    iteration_numbers = [250, 500]

    best_rmse = 10000000  # a big number
    best_params = {'det_encs': 0, 'lat_encs': 0, 'dec_hid': 0, 'lr': 0, 'batch_size': 0, 'iterations': 0}

    for det_encs in det_encoder_hidden_sizes:
        for lat_encs in lat_encoder_hidden_sizes:
            for dec_hid in decoder_hidden_sizes:
                for l_rate in learning_rates:
                    for batch_s in batch_sizes:
                        for iter_num in iteration_numbers:

                            m = AttentiveNP(x_size=X_train.shape[2], y_size=y_size, r_size=r_size,
                                            det_encoder_hidden_size=det_encs,
                                            det_encoder_n_hidden=det_encoder_n_hidden,
                                            lat_encoder_hidden_size=lat_encs,
                                            lat_encoder_n_hidden=lat_encoder_n_hidden,
                                            decoder_hidden_size=dec_hid,
                                            decoder_n_hidden=decoder_n_hidden,
                                            lr=l_rate, attention_type="multihead")

                            print('...training.')

                            m.train(X_train, y_train, batch_size=batch_s, iterations=iter_num, print_freq=None)

                            # Now, the context set comprises the training x / y values, the target set comprises the test x values.

                            y_pred, y_var = m.predict(X_train, y_train, X_test, n_samples=100)

                            # Output Standardised RMSE and RMSE on Train Set

                            score = r2_score(y_test, y_scaler.inverse_transform(y_pred))
                            rmse = np.sqrt(mean_squared_error(y_test, y_scaler.inverse_transform(y_pred)))
                            mae = mean_absolute_error(y_test, y_scaler.inverse_transform(y_pred))

                            print("\nR^2: {:.3f}".format(score))
                            print("RMSE: {:.3f}".format(rmse))
                            print("MAE: {:.3f}".format(mae))

                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_params['det_encs'] = det_encs
                                best_params['lat_encs'] = lat_encs
                                best_params['dec_hid'] = dec_hid
                                best_params['lr'] = l_rate
                                best_params['batch_size'] = batch_s
                                best_params['iterations'] = iter_num
                            print('Best parameters are \n')
                            print(best_params)

    print('Final best parameters are \n')
    print(best_params)

    with open(f'validation_hypers/{task}/ANP/hypers_{representation}.txt', 'w') as f:
        f.write(str(best_params))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task', default='FreeSolv',
                        help='Task name One of: [Photoswitch, ESOL, FreeSolv, Lipophilicity].')
    parser.add_argument('-p', '--path', default='../datasets/FreeSolv.csv',
                        help='Path to dataset csv file.')
    parser.add_argument('-r', '--representation', default='fragprints',
                        help='Descriptor type. One of [fingerprints, fragments, fragprints.')
    parser.add_argument('-pca', '--use_pca', type=bool, default=True,
                        help='If true, apply PCA to data (50 components).')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='Fraction of Dataset to use as test set.')
    parser.add_argument('-rs', '--r_size', type=int, default=8,
                        help='Dimensionality of context encoding, r.')
    parser.add_argument('-dnh', '--det_encoder_n_hidden', type=int, default=2,
                        help='Number of deterministic encoder hidden layers.')
    parser.add_argument('-lnh', '--lat_encoder_n_hidden', type=int, default=2,
                        help='Number of latent encoder hidden layers.')
    parser.add_argument('-denh', '--decoder_n_hidden', type=int, default=2,
                        help='Number of decoder hidden layers.')

    args = parser.parse_args()

    main(args.task, args.path, args.representation, args.use_pca, args.test_set_size, args.r_size,
         args.det_encoder_n_hidden, args.lat_encoder_n_hidden, args.decoder_n_hidden)
