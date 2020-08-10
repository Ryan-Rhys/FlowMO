"""
Script for creating enumerated random SMILES for the Photoswitch dataset.
"""

import numpy as np
import pandas as pd

from data_utils import TaskDataLoader
from smiles_x_enum import augmentation

if __name__ == '__main__':
    aug_factor = 15
    data_loader = TaskDataLoader('FreeSolv', '../datasets/FreeSolv.csv')
    smiles_list, y = data_loader.load_property_data()

    # Delete the longest molecule in the dataset (122 characters).

    # smiles_list.pop(50)
    # y = np.delete(y, 50)

    aug_smiles, smiles_card, properties = augmentation(np.array(smiles_list), y, aug_factor, canon=False, rotate=True)

    data_dict = {'SMILES': aug_smiles, 'expt': properties}

    df = pd.DataFrame(data=data_dict, columns=['SMILES', 'expt'])
    df.to_csv(f'enumerated_datasets/augmented_x{aug_factor}_FreeSolv')
