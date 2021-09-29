# Author: Henry Moss & Ryan-Rhys Griffiths
"""
Molecule kernels for Gaussian Process Regression implemented in GPflow.
"""

import sys

import grakel
import numpy.testing as npt
import pandas as pd
import pytest
import tensorflow as tf
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from GP.kernel_modules.random_walk import RandomWalk

sys.path.append('/Users/juliusschwartz/Mystuff/FlowMO')

@pytest.fixture
def load_data():
    test_dataset_file = '../../datasets/ESOL.csv'
    df = pd.read_csv(test_dataset_file)
    return df["smiles"].to_list()


@pytest.mark.parametrize(
    'weight, series_type, p',
    [
        (0.1, 'geometric', None),
        (0.1, 'exponential', None),
        #(0.3, 'geometric', None), TODO: Fix
        (0.5, 'exponential', None),
        #(0.2, 'geometric', 3), TODO: Implement
        #(0.8, 'exponential', 3), TODO: Implement
    ]
)
def test_random_walk_unlabelled(weight, series_type, p, load_data):
    adj_mats = [GetAdjacencyMatrix(MolFromSmiles(smiles)) for smiles in load_data[:50]]

    tensor_adj_mats = [tf.convert_to_tensor(adj_mat) for adj_mat in adj_mats]
    grakel_graphs = [grakel.Graph(adj_mat) for adj_mat in adj_mats]

    random_walk_grakel = grakel.kernels.RandomWalk(normalize=False, lamda=weight, kernel_type=series_type)
    grakel_results = random_walk_grakel.fit_transform(grakel_graphs)

    random_walk_FlowMo = RandomWalk(series_type=series_type, weight=weight, normalize=False)
    FlowMo_results = random_walk_FlowMo.K(tensor_adj_mats, tensor_adj_mats)

    import pdb; pdb.set_trace()

    npt.assert_almost_equal(
        grakel_results, FlowMo_results.numpy(),
        decimal=2
    )
