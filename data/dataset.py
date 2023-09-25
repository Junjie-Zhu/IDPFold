import os

import numpy as np
import torch
from torch.utils.data import Dataset


class BackboneDataset(Dataset):
    def __init__(self, data_dir = '../NMR_data/pkls'):
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

