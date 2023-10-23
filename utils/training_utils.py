import os
import datetime

import torch
import torch.distributed as dist



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
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url))
    torch.distributed.init_process_group(backend=args.dist_backend, timeout=datetime.timedelta(seconds=600),)
    torch.distributed.barrier()
