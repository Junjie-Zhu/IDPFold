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

def feature_from_pkl(pkl_dir, model_num=8):
    total_data = pickle.load(pkl_dir)

    # Sample multiple models from one sequence
    model_choice = np.random.choice(len(total_data), model_num)

    model_detail = []
    for idx in model_choice:
        res_serial = [res[0] for res in total_data[idx].keys()]
        atom_serial = [f'{residue}_{atom}'
                       for residue in res_serial
                       for atom in np.array(total_data[idx][residue][:, 0])]

        # Make one-hot embedding
        node_attr = torch.zeros([len(atom_serial), 60])
        for atoms in range(len(node_attr)):
            node_attr[atoms, atom_order[atoms]] = 1

        # Calculate distance along edges
        coords = [np.array(resc)[:, 1:] for resc in total_data[idx].values()]
        coords = torch.Tensor(coords).view(-1, 3)

        edge_attr = (coords.view(-1, 1, 3) - coords.view(1, -1, 3)).norm(dim=-1)

        model_detail.append([node_attr.numpy(), edge_attr.numpy()])



class BackboneDataset(Dataset):
    def __init__(self, data_dir = '../NMR_data/pkls'):
        self.data_list = os.listdir(data_dir)
