import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


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

        one_hot_atom = features.x
        one_hot_res = one_hot_atom
        for index in range(one_hot_atom.shape[0]):
            one_hot_res[index] = one_hot_atom[index] // 3

        features.x = one_hot_res.view(-1, 3)[:, 1].squeeze()
        features.pos = features.pos.view(-1, 3, 3)[:, 1, :].squeeze()
        features.z = torch.LongTensor(features.z.item() // 3)

        return features


def collate(batch):
    batch = [i for i in batch if i != None]
    
    if len(batch) == 0:
        return None

    features = {}
    for feature in batch[0]:
        
        
        if "name" not in feature:
            
            features[feature] = torch.stack([i[feature] for i in batch], dim=0)
            
            
        else:
            
            features[feature] = [i[feature] for i in batch]

    return features
