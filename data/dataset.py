import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

RESIDUE_LIST = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

ATOM_LIST = [f"{residue}_{atom}" for residue in RESIDUE_LIST for atom in ['N', 'CA', 'C']]
atom_order = {atom: i for i, atom in enumerate(ATOM_LIST)}


class BackboneDataset(Dataset):
    def __init__(self, data_dir='../NMR_data/processed', mode='train', split=0.8, max_length=768):
        self.data_dir = data_dir
        self.data_list = os.listdir(data_dir)
        self.max_length = max_length

        if mode == 'train':
            self.data_list = self.data_list[:int(split * len(self.data_list))]
        elif mode == 'val':
            self.data_list = self.data_list[int(split * len(self.data_list)) + 1:]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        features = torch.load(os.path.join(self.data_dir, self.data_list[index]))
        return features

