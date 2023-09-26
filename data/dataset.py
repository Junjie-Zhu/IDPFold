import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

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
    def __init__(self, data_dir='../NMR_data/pkls', mode='train', split=0.8):
        self.data_dir = data_dir
        self.data_list = os.listdir(data_dir)

        if mode == 'train':
            self.data_list = self.data_list[:int(split * len(self.data_list))]
        elif mode == 'val':
            self.data_list = self.data_list[int(split * len(self.data_list)) + 1:]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        with open(os.path.join(self.data_dir, self.data_list[index]), 'rb') as f:
            model = pickle.load(f)

        res_serial = [res for res in model.keys()]
        atom_serial = [f'{residue[0]}_{atom}'
                       for residue in res_serial
                       for atom in ['N', 'CA', 'C']]

        # Make one-hot embedding
        node_attr = torch.zeros([len(atom_serial), 60])
        for atoms in range(len(node_attr)):
            node_attr[atoms, atom_order[atom_serial[atoms]]] = 1

        coords = np.array([np.array(resc).astype(float) for resc in model.values()])

        features = {'node_attr': node_attr,
                    'coordinates': torch.Tensor(coords),
                    }
        return features


def collate(batch):
    batch = [i for i in batch if i is not None]

    if len(batch) == 0:
        return None

    features = {}
    for feature in batch[0]:
        features[feature] = torch.stack([i[feature] for i in batch], dim=0)

    return features
