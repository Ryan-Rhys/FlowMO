"""
Script for creating enumerated random SMILES for the Photoswitch dataset.
"""

import numpy as np
import pandas as pd

from data_utils import TaskDataLoader
from smiles_x_enum import augmentation

if __name__ == '__main__':
    data_loader = TaskDataLoader('Photoswitch', '../datasets/photoswitches.csv')
    smiles_list, y = data_loader.load_property_data()

    # Delete the longest molecule in the dataset (122 characters).

    smiles_list.pop(50)
    y = np.delete(y, 50)

    aug_smiles, smiles_card, properties = augmentation(np.array(smiles_list), y, canon=False, rotate=True)

    data_dict = {'SMILES': aug_smiles, 'E isomer pi-pi* wavelength in nm': properties}

    df = pd.DataFrame(data=data_dict, columns=['SMILES', 'E isomer pi-pi* wavelength in nm'])
    df.to_csv('enumerated_datasets/doubly_augmented_photoswitches')
