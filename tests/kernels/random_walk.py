# Author: Henry Moss & Ryan-Rhys Griffiths
"""
Molecule kernels for Gaussian Process Regression implemented in GPflow.
"""

import gpflow
import tensorflow as tf
import grakel
import pytest
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import sys

sys.path.append('/Users/juliusschwartz/Mystuff/FlowMO')

from GP.kernels.random_walk import RandomWalk

test_smiles = []
test_dataset_file = '../../smiles_enumeration/enumerated_datasets/ESOL/X_test_split_aug_x2_split_0.txt'
lines = open(test_dataset_file).readlines()
for line in open(test_dataset_file).readlines():
    test_smiles.append(line.strip('\n'))

adj_mats = [GetAdjacencyMatrix(MolFromSmiles(smiles)) for smiles in test_smiles]
tensor_adj_mats = [tf.convert_to_tensor(adj_mat) for adj_mat in adj_mats]
grakel_graphs = [grakel.Graph(adj_mat) for adj_mat in adj_mats]

random_walk_grakel = grakel.kernels.RandomWalk()
grakel_results = random_walk_grakel.fit_transform(grakel_graphs)
