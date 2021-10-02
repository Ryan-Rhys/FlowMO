# Author: Henry Moss & Ryan-Rhys Griffiths
"""
Molecule kernels for Gaussian Process Regression implemented in GPflow.
"""

import os

import grakel
import numpy.testing as npt
import pandas as pd
import pytest
import tensorflow as tf
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from GP.kernel_modules.random_walk import RandomWalk

@pytest.fixture
def load_data():
    benchmark_path = os.path.abspath(
        os.path.join(
            os.getcwd(), '..', '..', 'datasets', 'ESOL.csv'
        )
    )
    df = pd.read_csv(benchmark_path)
    smiles = df["smiles"].to_list()

    adj_mats = [GetAdjacencyMatrix(MolFromSmiles(smiles)) for smiles in smiles[:50]]
    tensor_adj_mats = [tf.convert_to_tensor(adj_mat) for adj_mat in adj_mats]
    grakel_graphs = [grakel.Graph(adj_mat) for adj_mat in adj_mats]

    return tensor_adj_mats, grakel_graphs


@pytest.mark.parametrize(
    'weight, series_type, p',
    [
        (0.1, 'geometric', None),
        (0.1, 'exponential', None),
        #(0.3, 'geometric', None), #NB: requires `method_type="baseline" in grakel kernel constructor
        (0.3, 'exponential', None),
        (0.3, 'geometric', 3),
        (0.8, 'exponential', 3),
    ]
)
def test_random_walk_unlabelled(weight, series_type, p, load_data):
    tensor_adj_mats, grakel_graphs = load_data

    random_walk_grakel = grakel.kernels.RandomWalk(normalize=True, lamda=weight, kernel_type=series_type, p=p)
    grakel_results = random_walk_grakel.fit_transform(grakel_graphs)

    random_walk_FlowMo = RandomWalk(p=p, series_type=series_type, weight=weight, normalize=True)
    FlowMo_results = random_walk_FlowMo.K(tensor_adj_mats, tensor_adj_mats)

    npt.assert_almost_equal(
        grakel_results, FlowMo_results.numpy(),
        decimal=2
    )
