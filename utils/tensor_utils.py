import torch
import numpy as np


def get_edge_indices(length, device):
    pair_indices = torch.cartesian_prod(torch.tensor(range(length)),
                                        torch.tensor(range(length))).transpose(-1, -2)

    row, col = pair_indices

    return row.to(device), col.to(device)


def get_extra_edge_indices(current_edges, current_num, full_num, batch_size, device):
    '''
    Adds extra edges when edges between a subset of atoms are already defined
    
    current_edges: current edge indices, size (N, 2)
    current num: the number of atoms between which edges are already defined
    full num: the total number of atoms

    '''
    pair_indices = torch.cartesian_prod(torch.tensor(range(current_num)),
                                        torch.tensor(range(full_num - current_num)) + current_num) \
        .transpose(-1, -2).repeat(batch_size, 1, 1)

    pair_indices = torch.cat((current_edges.transpose(-1, -2), pair_indices.to(device)), dim=2)
    row = pair_indices[:, 0, :]
    col = pair_indices[:, 1, :]

    return row.to(device), col.to(device)


def cdist(x_1, x_2):
    differences = x_1.unsqueeze(-2) - x_2.unsqueeze(-3)

    distances = torch.sqrt(torch.sum(torch.square(differences),
                                     dim=-1) + 1e-12)

    return differences, distances


def normal_prob(mean, std, z, mask):
    N = torch.sum(mask, dim=(-1, -2))
    logp = -N[:, :, None] / 2. * np.log(2 * np.pi * (std ** 2)) \
           - torch.sum(((z - mean) * mask) ** 2, dim=(-1)) / (2 * (std ** 2))

    return logp


def get_trace_computation_tensors(input_tensor):
    shape = input_tensor.shape

    mask = torch.eye(shape[-1] * shape[-2]).view(shape[-1] * shape[-2],
                                                 shape[-2], shape[-1])

    mask = torch.repeat_interleave(mask.unsqueeze(0), shape[1], dim=1).repeat(shape[0], 1, 1, 1).to(input_tensor.device)

    indices = torch.arange(0, shape[1]).repeat(shape[-1] * shape[-2]).to(input_tensor.device)

    return mask, indices
