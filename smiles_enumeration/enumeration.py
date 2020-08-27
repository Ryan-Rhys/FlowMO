"""
Script for creating enumerated random SMILES for the datasets.
"""

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from data_utils import TaskDataLoader
from smiles_x_enum import augmentation


def main(path, task, aug_factor, n_trials, test_set_size):
    """
    :param path: str specifying path to dataset.
    :param task: str specifying the task. One of ['Photoswitch', 'ESOL', 'FreeSolv', 'Lipophilicity']
    :param aug_factor: Factor by which to augment the SMILES dataset.
    :param n_trials: int specifying number of random train/test splits to use
    :param test_set_size: float in range [0, 1] specifying fraction of dataset to use as test set
    """
    data_loader = TaskDataLoader(task, path)
    smiles_list, y = data_loader.load_property_data()

    for i in range(0, n_trials):

        X_train, X_test, y_train, y_test = train_test_split(smiles_list, y, test_size=test_set_size, random_state=i)

        # Augment the train set SMILES by a factor equal to aug_factor

        X_train, smiles_card, y_train = augmentation(np.array(X_train), y_train, aug_factor, canon=False, rotate=True)

        # Augment the test set SMILES by a factor equal to aug_factor

        X_test_aug, smiles_test_card, y_test_aug = augmentation(np.array(X_test), y_test, aug_factor, canon=False, rotate=True)

        # Save the augmented train SMILES with fixed test set.

        np.savetxt(f'enumerated_datasets/{task}/X_train_split_aug_x{aug_factor}_split_{i}.txt', X_train, fmt="%s")
        np.savetxt(f'enumerated_datasets/{task}/X_test_split_aug_x{aug_factor}_split_{i}.txt', X_test, fmt="%s")
        np.savetxt(f'enumerated_datasets/{task}/y_train_split_aug_x{aug_factor}_split_{i}.txt', y_train)
        np.savetxt(f'enumerated_datasets/{task}/y_test_split_aug_x{aug_factor}_split_{i}.txt', y_test)

        # Save the augmented test SMILES. aug in front of filename denotes test set augmentation as well.

        np.savetxt(f'enumerated_datasets/{task}/aug_X_test_split_aug_x{aug_factor}_split_{i}.txt', X_test_aug, fmt="%s")
        np.savetxt(f'enumerated_datasets/{task}/aug_y_test_split_aug_x{aug_factor}_split_{i}.txt', y_test_aug)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='../datasets/FreeSolv.csv',
                        help='Path to the csv file for the task.')
    parser.add_argument('-t', '--task', type=str, default='FreeSolv',
                        help='str specifying the task. One of [Photoswitch, ESOL, FreeSolv, Lipophilicity].')
    parser.add_argument('-a', '--aug_factor', type=int, default=15,
                        help='Augmentation Factor.')
    parser.add_argument('-n', '--n_trials', type=int, default=20,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.2,
                        help='float in range [0, 1] specifying fraction of dataset to use as test set')

    args = parser.parse_args()

    main(args.path, args.task, args.aug_factor, args.n_trials, args.test_set_size)
