import torch
import numpy as np
import torch.nn as nn
from e3nn import o3

from model_utils import get_timestep_embedding
from instance_norm import EquivariantInstanceNorm
from graph_norm import EquivariantGraphNorm
from layer_norm import EquivariantLayerNormV2
from fast_layer_norm import EquivariantLayerNormFast


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


class GraphAttention(torch.nn.Module):
    '''
        1. Message = Alpha * Value
        2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
        3. 0e -> Activation -> Inner Product -> (Alpha)
        4. (0e+1e+...) -> (Value)
    '''

    def __init__(self,
                 irreps_node_input, irreps_node_attr,
                 irreps_edge_attr, irreps_node_output,
                 fc_neurons,
                 irreps_head, num_heads, irreps_pre_attn=None,
                 rescale_degree=False, nonlinear_message=False,
                 alpha_drop=0.1, proj_drop=0.1):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)

        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads)  # irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify()
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps('{}x0e'.format(mul_alpha))  # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()

        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(self.irreps_pre_attn,
                                         self.irreps_edge_attr, self.irreps_pre_attn, fc_neurons,
                                         use_activation=True, norm_layer=None, internal_weights=False)
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(self.irreps_pre_attn,
                                           self.irreps_edge_attr, irreps_attn_heads, fc_neurons=None,
                                           use_activation=False, norm_layer=None, internal_weights=True)
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps('{}x0e'.format(mul_alpha_head)),
                                                 num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(self.irreps_pre_attn,
                                     self.irreps_edge_attr, irreps_attn_all, fc_neurons,
                                     use_activation=False, norm_layer=None)
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(),
                num_heads)

        self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)),
                                    [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head)

        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot)  # Following GATv2

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_input,
                                                drop_prob=proj_drop)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars,
                batch, **kwargs):

        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]

        if self.nonlinear_message:
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars)
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))

        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)

        if self.rescale_degree:
            degree = torch_geometric.utils.degree(edge_dst,
                                                  num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree = degree.view(-1, 1)
            attn = attn * degree

        node_output = self.proj(attn)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str

    
class TransBlock(torch.nn.Module):
    '''
        1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
        2. Use pre-norm architecture
    '''

    def __init__(self,
                 irreps_node_input, irreps_node_attr,
                 irreps_edge_attr, irreps_node_output,
                 fc_neurons,
                 irreps_head, num_heads, irreps_pre_attn=None,
                 rescale_degree=False, nonlinear_message=False,
                 alpha_drop=0.1, proj_drop=0.1,
                 drop_path_rate=0.0,
                 irreps_mlp_mid=None,
                 norm_layer='layer'):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input

        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ga = GraphAttention(irreps_node_input=self.irreps_node_input,
                                 irreps_node_attr=self.irreps_node_attr,
                                 irreps_edge_attr=self.irreps_edge_attr,
                                 irreps_node_output=self.irreps_node_input,
                                 fc_neurons=fc_neurons,
                                 irreps_head=self.irreps_head,
                                 num_heads=self.num_heads,
                                 irreps_pre_attn=self.irreps_pre_attn,
                                 rescale_degree=self.rescale_degree,
                                 nonlinear_message=self.nonlinear_message,
                                 alpha_drop=alpha_drop,
                                 proj_drop=proj_drop)

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None

        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)
        # self.concat_norm_output = ConcatIrrepsTensor(self.irreps_node_input,
        #    self.irreps_node_input)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,  # self.concat_norm_output.irreps_out,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop)
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input, self.irreps_node_attr,
                self.irreps_node_output,
                bias=True, rescale=_RESCALE)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars,
                batch, **kwargs):

        node_output = node_input
        node_features = node_input
        node_features = self.norm_1(node_features, batch=batch)
        # norm_1_output = node_features
        node_features = self.ga(node_input=node_features,
                                node_attr=node_attr,
                                edge_src=edge_src, edge_dst=edge_dst,
                                edge_attr=edge_attr, edge_scalars=edge_scalars,
                                batch=batch)

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        node_features = node_output
        node_features = self.norm_2(node_features, batch=batch)
        # node_features = self.concat_norm_output(norm_1_output, node_features)
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        return node_output


class Siege(nn.Module):

    def __init__(self, config):
        super(Siege, self).__init__()

        self.h_a = config['h_a']
        self.h_b = config['h_b']
        self.h_t = config['h_t']
        self.n_conv = config['n_conv']

        self.node_in = nn.Embedding(60, self.h_a)
        self.t_embed_in = get_timestep_embedding('sinusoidal', embedding_dim=self.h_t)

        self.convs = nn.ModuleList([ConvLayer(self.h_a + self.h_t, self.h_b, random_seed=999)
                                    for _ in range(self.n_conv)])
        self.e_out = nn.Linear(self.h_a, 1)

    def forward(self, node_attr, edge_attr, edge_idx, t, atom_mask):

        node_attr = torch.cat([self.node_in(node_attr), self.t_embed_in(t)], 1)

        for idx in range(self.n_conv):
            node_attr = node_attr * atom_mask

            # Change network settings
            node_attr = self.convs[idx](node_attr, edge_attr, edge_idx, atom_mask)

        node_attr = node_attr * atom_mask
        e_out = torch.sum(self.e_out(node_attr))

        return e_out

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

    @staticmethod
    def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)

        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF


def randomSeed(random_seed):
    """Given a random seed, this will help reproduce results across runs"""
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
