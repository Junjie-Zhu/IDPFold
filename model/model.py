import torch
import numpy as np
import torch.nn as nn
from e3nn import o3
from torch_cluster import radius_graph

from equiformer.drop import EquivariantDropout
from equiformer.instance_norm import EquivariantInstanceNorm
from equiformer.graph_norm import EquivariantGraphNorm
from equiformer.layer_norm import EquivariantLayerNormV2
from equiformer.fast_layer_norm import EquivariantLayerNormFast
from equiformer.graph_attention_transformer import (TransBlock, NodeEmbeddingNetwork,
                                                    EdgeDegreeEmbeddingNetwork, ScaledScatter)
from equiformer.gaussian_rbf import GaussianRadialBasisLayer
from equiformer.tensor_product_rescale import LinearRS
from equiformer.fast_activation import Activation

from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis

_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 60

_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666


def get_norm_layer(norm_type):
    if norm_type == 'graph':
        return EquivariantGraphNorm
    elif norm_type == 'instance':
        return EquivariantInstanceNorm
    elif norm_type == 'layer':
        return EquivariantLayerNormV2
    elif norm_type == 'fast_layer':
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError('Norm type {} not supported.'.format(norm_type))


class Siege(nn.Module):

    def __init__(self,
                 irreps_in='5x0e',
                 irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                 irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                 max_radius=5.0,
                 number_of_basis=128, basis_type='gaussian', fc_neurons=[64, 64],
                 irreps_feature='512x0e',
                 irreps_head='32x0e+16x1o+8x2e', num_heads=4, irreps_pre_attn=None,
                 rescale_degree=False, nonlinear_message=False,
                 irreps_mlp_mid='128x0e+64x1e+32x2e',
                 norm_layer='layer',
                 alpha_drop=0.2, proj_drop=0.0, out_drop=0.0,
                 drop_path_rate=0.0,
                 mean=None, std=None, scale=None, atomref=None
                 ):
        super(Siege, self).__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer('atomref', atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None \
            else o3.Irreps.spherical_harmonics(self.lmax)
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)

        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius,
                                   rbf={'name': 'spherical_bessel'})
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding,
                                                         self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE)

        self.blocks = torch.nn.ModuleList()
        self.build_blocks()

        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE),
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps('1x0e'), rescale=_RESCALE))
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.apply(self._init_weights)

    def build(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding,
                             irreps_node_attr=self.irreps_node_attr,
                             irreps_edge_attr=self.irreps_edge_attr,
                             irreps_node_output=irreps_block_output,
                             fc_neurons=self.fc_neurons,
                             irreps_head=self.irreps_head,
                             num_heads=self.num_heads,
                             irreps_pre_attn=self.irreps_pre_attn,
                             rescale_degree=self.rescale_degree,
                             nonlinear_message=self.nonlinear_message,
                             alpha_drop=self.alpha_drop,
                             proj_drop=self.proj_drop,
                             drop_path_rate=self.drop_path_rate,
                             irreps_mlp_mid=self.irreps_mlp_mid,
                             norm_layer=self.norm_layer)
            self.blocks.append(blk)

    def forward(self, f_in, pos, batch, node_atom, **kwargs) -> torch.Tensor:

        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch,
                                          max_num_neighbors=1000)
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
                                         x=edge_vec, normalize=True, normalization='component')

        atom_embedding, atom_attr, atom_onehot = self.atom_embed(f_in)
        edge_length = edge_vec.norm(dim=1)

        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh,
                                                    edge_length_embedding, edge_src, edge_dst, batch)
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        for blk in self.blocks:
            node_features = blk(node_input=node_features, node_attr=node_attr,
                                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh,
                                edge_scalars=edge_length_embedding,
                                batch=batch)

        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
        outputs = self.head(node_features)
        outputs = self.scale_scatter(outputs, batch, dim=0)

        if self.scale is not None:
            outputs = self.scale * outputs

        return outputs

    def predict(self, node_attr, coords, t, atom_mask):
        B, N, C = coords.shape[:]

        coords = coords * atom_mask

        edge_attr = (coords.unsqueeze(dim=-2) - coords.unsqueeze(dim=-3)).norm(dim=-1)
        non_self_mask = ~torch.eye(N, dtype=torch.bool)
        non_self_mask = non_self_mask.unsqueeze(0).expand(B, -1, -1)

        edge_attr = edge_attr[non_self_mask].view(B, N, N - 1, 1)

        edge_idx = torch.LongTensor(torch.arange(N).expand(N, -1).unsqueeze(0).expand(B, -1, -1))
        edge_idx = edge_idx[non_self_mask].view(B, N, N - 1)
        e_out = self.forward(node_attr, edge_attr, edge_idx, t, atom_mask)
        return e_out

    def predict_forces(self, node_attr, coords, t, sde, atom_mask):
        _, std = sde.marginal_prob(torch.zeros(coords.shape).to(coords.device), t)

        coords = coords.clone().requires_grad_()
        self.predict(node_attr, coords, t, atom_mask).backward()
        forces = - 1 * coords.grad / std[:, None, None, None]

        return forces
