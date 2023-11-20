import os
import datetime
from typing import Union, List

import torch
import torch.distributed as dist



def sample_noise(sde, x, batch, device, eps=1e-5):
    t_concat = torch.rand(batch[-1].item() + 1) * (sde.T - eps) + eps
    t = torch.zeros(x.shape[0])

    for i in range(x.shape[0]):
        t[i] = t_concat[batch[i]]

    z = torch.randn_like(x)

    mean, std = sde.marginal_prob(x, t)
    
    mean = torch.FloatTensor(mean)
    std = torch.FloatTensor(std).unsqueeze(-1)

    perturbed_data = mean + torch.mul(z, std)

    return z.to(device), t.to(device), \
        perturbed_data.to(device), \
        mean.to(device), std.to(device)


def dsm(prediction, std, z):
    all_losses = torch.square(torch.mul(std, prediction) + z)

    loss = torch.mean(torch.sum(all_losses, dim=-1))

    return all_losses, loss


def setup(backend='nccl'):
    """ 初始化DDP """
    torch.distributed.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=3600))
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    return local_rank, device


def sync_tensor_across_gpus(t: Union[torch.Tensor, List, None]
                            ) -> Union[torch.Tensor, None]:
    """ 用来收集来自于多个进程的结果 """
    # t needs to have dim 0 for troch.cat below.
    # if not, you need to prepare it.
    if t is None:
        return None
    if isinstance(t, List):
        t = torch.stack(t)
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t)  # this works with nccl backend when tensors need to be on gpu.
    # for gloo and mpi backends, tensors need to be on cpu. also this works single machine with
    # multiple   gpus. for multiple nodes, you should use dist.all_gather_multigpu. both have the
    # same definition... see [here](https://pytorch.org/docs/stable/distributed.html).
    #  somewhere in the same page, it was mentioned that dist.all_gather_multigpu is more for
    # multi-nodes. still dont see the benefit of all_gather_multigpu. the provided working case in
    # the doc is  vague...
    return torch.cat(gather_t_tensor, dim=0)