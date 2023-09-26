from torch_scatter import scatter, scatter_add
from tensor_utils import normal_prob
import torch


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
