import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


def sample_noise(sde, x, device, eps=1e-5):
    t = torch.rand(x.shape[0]) * (sde.T - eps) + eps

    z = torch.randn_like(x)

    mean, std = sde.marginal_prob(x, t)

    mean = torch.FloatTensor(mean)
    std = torch.FloatTensor(std)

    perturbed_data = mean + std[:, None, None, None] * z

    return z.to(device), t.to(device), \
        perturbed_data.to(device), \
        mean.to(device), std.to(device)


def dsm(prediction, std, z):
    all_losses = torch.square(prediction * std[:, None, None, None] + z)

    loss = torch.mean(torch.sum(all_losses, dim=(-1, -2, -3)))

    return all_losses, loss


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x: sinusoidal_embedding(x, embedding_dim, max_positions=embedding_scale))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2)).float()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    #print('| distributed init (rank {}): {}'.format(
    #    args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
