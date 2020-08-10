# Author: Ryan-Rhys Griffiths
"""
Module containing data loading utility functions.
"""

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TaskDataLoader:
    """
    Data loader class
    """

    def __init__(self, task, path):
        """
        :param task: Property prediction task to load data from:
        ['Photoswitch', 'ESOL', 'FreeSolv', 'Lipophilicity']
        :param path: Path to the corresponding csv file for the task
        """

        self.task = task
        self.path = path

    def load_property_data(self):
        """
        Load data corresponding to the property prediction task.
        :return: smiles_list, property_vals
        """

        df = pd.read_csv(self.path)

        if self.task == 'Photoswitch':

            # Load the SMILES as x-values and the E isomer pi-pi* wavelength in nm as the y-values.
            smiles_list = df['SMILES'].to_list()
            property_vals = df['E isomer pi-pi* wavelength in nm'].to_numpy()

        elif self.task == 'ESOL':
            smiles_list = df['smiles'].tolist()
            property_vals = df['measured log solubility in mols per litre'].to_numpy()

        elif self.task == 'FreeSolv':
            smiles_list = df['smiles'].tolist()
            property_vals = df['expt'].to_numpy()  # can change to df['calc'] for calculated values

        elif self.task == 'Lipophilicity':
            smiles_list = df['smiles'].tolist()
            property_vals = df['exp'].to_numpy()

        else:
            raise Exception('Must specify a valid task')

        # delete SMILES entries where the corresponding property values is NAN.

        smiles_list = list(np.delete(np.array(smiles_list), np.argwhere(np.isnan(property_vals))))
        property_vals = np.delete(property_vals, np.argwhere(np.isnan(property_vals)))

        return smiles_list, property_vals


def transform_data(X_train, y_train, X_test, y_test, n_components=None, use_pca=False):
    """
    Apply feature scaling, dimensionality reduction to the data. Return the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: input train data
    :param y_train: train labels
    :param X_test: input test data
    :param y_test: test labels
    :param n_components: number of principal components to keep when use_pca = True
    :param use_pca: Whether or not to use PCA
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    """

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    if use_pca:
        pca = PCA(n_components)
        X_train_scaled = pca.fit_transform(X_train)
        print('Fraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))
        X_test_scaled = pca.transform(X_test)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler


def featurise_mols(smiles_list, representation, bond_radius=3, nBits=2048):
    """
    Featurise molecules according to representation
    :param smiles_list: list of molecule SMILES
    :param representation: str giving the representation. Can be 'fingerprints' or 'fragments'.
    :param bond_radius: int giving the bond radius for Morgan fingerprints. Default is 3
    :param nBits: int giving the bit vector length for Morgan fingerprints. Default is 2048
    :return: X, the featurised molecules
    """

    if representation == 'fingerprints':

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        X = [AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=nBits) for mol in rdkit_mols]
        X = np.asarray(X)

    elif representation == 'fragments':

        # descList[115:] contains fragment-based features only
        # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)

        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        X = np.zeros((len(smiles_list), len(fragments)))
        for i in range(len(smiles_list)):
            mol = MolFromSmiles(smiles_list[i])
            try:
                features = [fragments[d](mol) for d in fragments]
            except:
                raise Exception('molecule {}'.format(i) + ' is not canonicalised')
            X[i, :] = features

    elif representation == 'fragprints':

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        X = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048) for mol in rdkit_mols]
        X = np.asarray(X)

        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        X1 = np.zeros((len(smiles_list), len(fragments)))
        for i in range(len(smiles_list)):
            mol = MolFromSmiles(smiles_list[i])
            try:
                features = [fragments[d](mol) for d in fragments]
            except:
                raise Exception('molecule {}'.format(i) + ' is not canonicalised')
            X1[i, :] = features

        X = np.concatenate((X, X1), axis=1)

    else:

        # SMILES

        return smiles_list

    return X
