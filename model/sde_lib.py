"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs. 
Adapted from source code at https://github.com/yang-song/score_sde_pytorch
"""
from torch_scatter import scatter
import abc
import torch
import numpy as np


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, means=None):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
        pass

    def discretize(self, x, t, means=None):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t, means)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, means=None):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t, means)

                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)

                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t, means):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, means=means)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000,
                 temperature=1):
        """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.temperature = temperature

    @property
    def T(self):
        return 1

    def sde(self, x, t, means=None):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x

        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x

        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.temperature

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t, means=None):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x

        G = sqrt_beta
        return f, G


class cmVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000, temperature=1):
        """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.temperature = temperature

    @property
    def T(self):
        return 1

    def sde(self, x, t, means=None):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x + 0.5 * beta_t[:, None, None, None] * means
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t, atom_mask, indices=None):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x

        if indices == None:

            centroids = torch.sum(x * atom_mask[:, None, :, :],
                                  dim=2)[:, :, None, :] / torch.sum(atom_mask, dim=1)[:, None, None, :]

        else:

            centroids = scatter(src=x * atom_mask[:, None, :, :], dim=2,
                                index=indices.unsqueeze(1),
                                reduce="sum")
            centroids = centroids / (scatter(src=atom_mask[:, None, :, :], dim=2,
                                             index=indices.unsqueeze(1), reduce="sum") + 1e-8)
            centroids = torch.stack([centroids[i][:, indices[i], :] \
                                     for i in range(len(indices))])

        centroids = centroids * atom_mask[:, None, :, :]

        mean = mean + (1 - torch.exp(log_mean_coeff[:, None, None, None])) * centroids

        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

        return mean, std

    def prior_sampling(self, shape, atom_mask, means, indices=None):

        z = torch.randn(*shape).to(means.device) * self.temperature

        if indices == None:

            centroids = torch.sum(z * atom_mask[:, None, :, :],
                                  dim=2)[:, :, None, :] / torch.sum(atom_mask, dim=1)[:, None, None, :]

        else:

            centroids = scatter(src=z.to(means.device) * atom_mask[:, None, :, :], dim=2,
                                index=indices.unsqueeze(1),
                                reduce="sum")
            centroids = centroids / (scatter(src=atom_mask[:, None, :, :], dim=2,
                                             index=indices.unsqueeze(1), reduce="sum") + 1e-8)
            centroids = torch.stack([centroids[i][:, indices[i], :] \
                                     for i in range(len(indices))])

        z = z - centroids
        z = (z + means) * atom_mask[:, None, :, :]
        return z

    def prior_logp(self, z, num_centers, mask):

        N = torch.sum(mask, dim=(1, 2)).unsqueeze(-1)

        logps = -(N - num_centers * 3) / 2. * np.log(2 * np.pi * (self.temperature ** 2)) \
                - torch.sum((z * mask) ** 2, dim=(2, 3)) / (2 * (self.temperature ** 2))

        return logps


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000,
                 temperature=1, mask_first_atom=False):
        """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.temperature = temperature
        self.mask_first_atom = mask_first_atom

    @property
    def T(self):
        return 1

    def sde(self, x, t, means=None):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)

        drift = -0.5 * torch.mul(beta_t.unsqueeze(-1), x)

        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        mean = torch.mul(torch.exp(log_mean_coeff).unsqueeze(-1), x)

        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.temperature

    def prior_logp(self, z, mask):
        if self.mask_first_atom:
            mask[:, 0, :] = 0

        N = torch.sum(mask, dim=(1, 2)).unsqueeze(-1)
        return -N / 2. * np.log(2 * np.pi * (self.temperature ** 2)) \
               - torch.sum((z * mask[:, None, :, :]) ** 2, dim=(2, 3)) / (2 * (self.temperature ** 2))


class cmsubVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000,
                 temperature=1):
        """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.temperature = temperature

    @property
    def T(self):
        return 1

    def sde(self, x, t, means=None):

        t = torch.clamp(t, min=0)

        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)

        drift = -0.5 * beta_t[:, None, None, None] * x + 0.5 * beta_t[:, None, None, None] * means

        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)

        return drift, diffusion

    def marginal_prob(self, x, t, atom_mask, indices=None, center_mask=None):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x

        if indices == None:

            if center_mask == None:
                centroids = torch.sum(x * atom_mask[:, None, :, :],
                                      dim=2)[:, :, None, :] / torch.sum(atom_mask, dim=1)[:, None, None, :]
            else:
                centroids = torch.sum(x * atom_mask[:, None, :, :] * center_mask[:, None, :, None],
                                      dim=2)[:, :, None, :] / torch.sum(atom_mask, dim=1)[:, None, None, :]


        else:

            centroids = scatter(src=x * atom_mask[:, None, :, :], dim=2,
                                index=indices.unsqueeze(1),
                                reduce="sum")
            centroids = centroids / (scatter(src=atom_mask[:, None, :, :], dim=2,
                                             index=indices.unsqueeze(1), reduce="sum") + 1e-8)
            centroids = torch.stack([centroids[i][:, indices[i], :] \
                                     for i in range(len(indices))])
        centroids = centroids * atom_mask[:, None, :, :]

        if center_mask == None:
            mean = mean + (1 - torch.exp(log_mean_coeff[:, None, None, None])) * centroids
        else:
            mean = (mean + (1 - torch.exp(log_mean_coeff[:, None, None, None])) * centroids) \
                   * center_mask[:, None, :, None] + (1 - center_mask[:, None, :, None]) * mean

        std = 1 - torch.exp(2. * log_mean_coeff)

        return mean, std

    def prior_sampling(self, shape, atom_mask, means, indices=None):

        z = torch.randn(*shape).to(means.device) * self.temperature

        if indices == None:

            centroids = torch.sum(z * atom_mask[:, None, :, :],
                                  dim=2)[:, :, None, :] / torch.sum(atom_mask, dim=1)[:, None, None, :]

        else:

            centroids = scatter(src=z.to(means.device) * atom_mask[:, None, :, :], dim=2,
                                index=indices.unsqueeze(1),
                                reduce="sum")
            centroids = centroids / (scatter(src=atom_mask[:, None, :, :], dim=2,
                                             index=indices.unsqueeze(1), reduce="sum") + 1e-8)
            centroids = torch.stack([centroids[i][:, indices[i], :] \
                                     for i in range(len(indices))])
        z = z - centroids

        z_init = z.clone()

        z = (z + means) * atom_mask[:, None, :, :]

        return z, z_init

    def prior_logp(self, z, num_centers, mask):
        N = torch.sum(mask, dim=(1, 2)).unsqueeze(-1)
        return -(N - num_centers * 3) / 2. * np.log(2 * np.pi * (self.temperature ** 2)) \
               - torch.sum((z * mask) ** 2, dim=(2, 3)) / (2 * (self.temperature ** 2))


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t, means=None):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (
                2 * self.sigma_max ** 2)

    def discretize(self, x, t, means=None):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G
