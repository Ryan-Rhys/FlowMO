# Author: Henry Moss & Ryan-Rhys Griffiths
"""
Verifies the FlowMO implementation of the Random Walk graph kernel
against GraKel
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
from GP.kernel_modules.kernel_utils import preprocess_adjacency_matrix_inputs

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
    preprocessed_tensor_adj_mats = preprocess_adjacency_matrix_inputs(tensor_adj_mats)
    grakel_graphs = [grakel.Graph(adj_mat) for adj_mat in adj_mats]

    return preprocessed_tensor_adj_mats, grakel_graphs


@pytest.mark.parametrize(
    'weight, series_type, p',
    [
        (0.1, 'geometric', None),
        (0.1, 'exponential', None),
        #(0.3, 'geometric', None), #Requires `method_type="baseline" in GraKel kernel constructor
        (0.3, 'exponential', None),
        (0.3, 'geometric', 3), #Doesn't pass due to suspected GraKel bug, see https://github.com/ysig/GraKeL/issues/71
        (0.8, 'exponential', 3), #Same issue as above test
    ]
)
def test_random_walk_unlabelled(weight, series_type, p, load_data):
    preprocessed_tensor_adj_mats, grakel_graphs = load_data

    random_walk_grakel = grakel.kernels.RandomWalk(normalize=True, lamda=weight, kernel_type=series_type, p=p)
    grakel_results = random_walk_grakel.fit_transform(grakel_graphs)

    random_walk_FlowMo = RandomWalk(normalize=True, weight=weight, series_type=series_type, p=p)
    FlowMo_results = random_walk_FlowMo.K(preprocessed_tensor_adj_mats, preprocessed_tensor_adj_mats)

    npt.assert_almost_equal(
        grakel_results, FlowMo_results.numpy(),
        decimal=2
    )
