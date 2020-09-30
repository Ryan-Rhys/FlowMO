# GP-Molecule
Library for training Gaussian Processes on Molecules

## Install

We recommend using a conda environment.

``conda create -n gp_molecule python==3.7``

```
conda install -c conda-forge rdkit
conda install matplotlib pytest scikit-learn pandas pytorch
pip install git+https://github.com/GPflow/GPflow.git@develop#egg=gpflow

cd Theano-master
python setup.py install
```

