import torch
import numpy as np
import torch.nn as nn
from e3nn import o3


class ConvLayer(nn.Module):

    def __init__(self, h_a, h_b, random_seed=None):
        randomSeed(random_seed)
        super(ConvLayer, self).__init__()
        self.h_a = h_a
        self.h_b = h_b
        self.fc_full = nn.Linear(2 * self.h_a + self.h_b, 2 * self.h_a)
        self.sigmoid = nn.Sigmoid()
        self.activation_hidden = nn.ReLU()
        self.bn_hidden = nn.BatchNorm1d(2 * self.h_a)
        self.bn_output = nn.BatchNorm1d(self.h_a)
        self.activation_output = nn.ReLU()

    def forward(self, atom_emb, nbr_emb, nbr_adj_list):
        N, M = nbr_adj_list.shape[1:]
        B = atom_emb.shape[0]

        atom_nbr_emb = atom_emb[torch.arange(B).unsqueeze(-1), nbr_adj_list.view(B, -1)].view(B, N, M, self.h_a)

        total_nbr_emb = torch.cat([atom_emb.unsqueeze(2).expand(B, N, M, self.h_a), atom_nbr_emb, nbr_emb], dim=-1)
        total_gated_emb = self.fc_full(total_nbr_emb)
        total_gated_emb = self.bn_hidden(total_gated_emb.view(-1, self.h_a * 2)).view(B, N, M, self.h_a * 2)
        nbr_filter, nbr_core = total_gated_emb.chunk(2, dim=3)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.activation_hidden(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=2)
        nbr_sumed = self.bn_output(nbr_sumed.view(-1, self.h_a)).view(B, N, self.h_a)
        out = self.activation_output(atom_emb + nbr_sumed)

        return out


class IDPFold(nn.Module):

    def __init__(self, **kwargs):
        super(IDPFold, self).__init__()

        self.h_a = kwargs.get('h_a', 32)
        self.h_b = kwargs.get('h_b', 1)
        self.n_conv = kwargs.get('n_conv', 2)
        # self.embedding = nn.Embedding(60, 32)
        self.node_in = nn.Linear(1, 32)

        self.convs = nn.ModuleList([ConvLayer(self.h_a, self.h_b, random_seed=999) for _ in range(self.n_conv)])
        self.e_out = nn.Linear(32, 1)

    def forward(self, node_attr, edge_attr, edge_idx):
        node_attr = self.node_in(node_attr)
        
        for idx in range(self.n_conv):
            node_attr = self.convs[idx](node_attr, edge_attr, edge_idx)

        e_out = torch.sum(self.e_out(node_attr))

        return e_out


def predict(coords):
    node_attr = np.arange(3)
    node_attr = torch.Tensor(node_attr).unsqueeze(-1).unsqueeze(0)

    edge_attr = (coords.view(-1, 1, 3) - coords.view(1, -1, 3)).norm(dim=-1)
    non_self_mask = edge_attr != 0
    edge_attr = edge_attr[non_self_mask].view(1, -1, 2, 1)

    edge_idx = torch.LongTensor([[[1, 2], [0, 2], [0, 1]]])
    
    model = IDPFold()
    e_out = model(node_attr, edge_attr, edge_idx)
    return e_out


def predict_forces(coords):
    coords = coords.clone().requires_grad_()
    predict(coords).backward()
    forces = -coords.grad

    return forces


def randomSeed(random_seed):
    """Given a random seed, this will help reproduce results across runs"""
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


coords = torch.randn(3, 3)
rot = o3.rand_matrix()

# 预测的势能是不变的，以下式子应当成立：predict(coords) == predict(coords @ rot)
print('predict(coords)\n', predict(coords), sep='')
print('predict(coords @ rot)\n', predict(coords @ rot), sep='')

# 预测的力是等变的，以下式子应当成立：predict_forces(coords) @ rot == predict_forces(coords @ rot)
print('predict_forces(coords) @ rot\n', predict_forces(coords) @ rot, sep='')
print('predict_forces(coords @ rot)\n', predict_forces(coords @ rot), sep='')

