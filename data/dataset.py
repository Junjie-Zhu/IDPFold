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
    def __init__(self, data_dir='../NMR_data/pkls', mode='train', split=0.8, max_length=768):
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
        name = self.data_list[index]
        with open(os.path.join(self.data_dir, self.data_list[index]), 'rb') as f:
            model = pickle.load(f)

        res_serial = [res for res in model.keys() if len(model[res]) == 3]
        atom_serial = [f'{residue[0]}_{atom}'
                       for residue in res_serial
                       for atom in ['N', 'CA', 'C']]

        if len(atom_serial) > self.max_length:  # Select sequences of length < 256
            return None

        z = torch.tensor(len(atom_serial), dtype=torch.long)

        # Make one-hot embedding
        node_attr = torch.zeros([len(atom_serial)], dtype=torch.long)

        for atoms in range(len(node_attr)):
            node_attr[atoms] = atom_order[atom_serial[atoms]]

        # Get coordinates for diffusion and edge attribution
        coords = np.array([np.array(resc).astype(float) for resc in model.values() if len(resc) == 3])
        coords = torch.Tensor(coords).view(-1, 3)
        centered_coords = coords - torch.sum(coords, dim=0)[None, :] / len(coords)

        node_index = torch.tensor([i for i in range(len(atom_serial))])
        edge_d_dst_index = torch.repeat_interleave(node_index, repeats=len(atom_serial))
        edge_d_src_index = node_index.repeat(len(atom_serial))
        edge_d_attr = centered_coords[edge_d_dst_index] - centered_coords[edge_d_src_index]
        edge_d_attr = edge_d_attr.norm(dim=1, p=2)

        edge_d_dst_index = edge_d_dst_index.view(1, -1)
        edge_d_src_index = edge_d_src_index.view(1, -1)
        edge_d_index = torch.cat((edge_d_dst_index, edge_d_src_index), dim=0)

        data = Data(x=node_attr, pos=centered_coords, z=z, name=name,
                    edge_d_index=edge_d_index, edge_d_attr=edge_d_attr)

        features = {'node_attr': node_attr,
                    'coordinates': centered_coords,
                    'atom_mask': atom_mask.unsqueeze(-1),
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
