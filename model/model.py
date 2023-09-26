import torch
import numpy as np
import torch.nn as nn


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


class Siege(nn.Module):

    def __init__(self, config):
        super(Siege, self).__init__()

        self.h_a = config['h_a']
        self.h_b = config['h_b']
        self.n_conv = config['n_conv']
        self.node_in = nn.Embedding(60, self.h_a)

        self.convs = nn.ModuleList([ConvLayer(self.h_a, self.h_b, random_seed=999) for _ in range(self.n_conv)])
        self.t_embed = nn.ModuleList([TimeLinear(self.h_a) for _ in range(self.n_conv)])
        self.e_out = nn.Linear(128, 1)

    def forward(self, node_attr, edge_attr, edge_idx, t):
        node_attr = self.node_in(node_attr)

        for idx in range(self.n_conv):
            node_attr = self.convs[idx](node_attr, edge_attr, edge_idx)
            node_attr = self.t_embed[idx](node_attr, t)

        e_out = torch.sum(self.e_out(node_attr))

        return e_out

    def predict(self, node_attr, coords, t):
        B, N, C = coords.shape()
        edge_attr = (coords.unsqueeze(dim=-1) - coords.unsqueeze(dim=-2)).norm(dim=-1)
        non_self_mask = edge_attr != 0
        edge_attr = edge_attr[non_self_mask].view(B, N, N - 1, 1)

        edge_idx = torch.LongTensor(torch.arange(N).expand(N, -1))
        diagonal_mask = ~torch.eye(edge_idx.size(0), dtype=torch.bool)
        edge_idx = edge_idx[diagonal_mask].view(B, N, N - 1)

        e_out = self.forward(node_attr, edge_attr, edge_idx, t)

        return e_out

    def predict_forces(self, node_attr, coords, t, sde):
        _, std = sde.marginal_prob(torch.zeros(coords.shape).to(coords.device), t)

        coords = coords.clone().requires_grad_()
        self.predict(node_attr, coords, t).backward()
        forces = - 1 * coords.grad / std[:, None, None, None]

        return forces


class TimeLinear(nn.Module):

    def __init__(self, h_a):
        super(TimeLinear, self).__init__()

        self.time_weight_linear = nn.Linear(1, h_a, bias=False)
        self.time_bias_linear = nn.Linear(1, h_a, bias=False)

    def forward(self, node_attr, t):
        t = t.view(node_attr.shape[0], 1, 1)

        node_attr = node_attr * torch.Sigmoid(self.time_weight_linear(t)) + torch.tanh(self.time_bias_linear(t))

        return node_attr


def randomSeed(random_seed):
    """Given a random seed, this will help reproduce results across runs"""
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
