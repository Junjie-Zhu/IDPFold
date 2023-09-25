# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from torch_scatter import scatter, scatter_add
from torchdiffeq import odeint_adjoint as odeint
from scipy import integrate

import sde_lib
from ..utils.tensor_utils import get_trace_computation_tensors

_CORRECTORS = {}
_PREDICTORS = {}


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, atom_mask=None, indices=None, means=None,
                    likelihood=False):
    """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':

        if not likelihood:
            sampling_fn = get_ode_sampler(sde=sde,
                                          shape=shape,
                                          inverse_scaler=inverse_scaler,
                                          denoise=config.sampling.noise_removal,
                                          rtol=config.sampling.rtol,
                                          atol=config.sampling.atol,
                                          eps=eps,
                                          device=config.device,
                                          atom_mask=atom_mask,
                                          indices=indices,
                                          means=means)
        else:
            sampling_fn = get_sampling_likelihood_fn(sde=sde,
                                                     shape=shape,
                                                     inverse_scaler=inverse_scaler,
                                                     rtol=config.sampling.rtol,
                                                     atol=config.sampling.atol,
                                                     eps=eps,
                                                     device=config.device,
                                                     atom_mask=atom_mask,
                                                     indices=indices,
                                                     means=means)

    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device,
                                     atom_mask=atom_mask,
                                     indices=indices,
                                     means=means)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, atom_mask=None, indices=None):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, atom_mask=None, indices=None):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, atom_mask=None, indices=None, means=None):

        dt = -1. / self.rsde.N
        z = torch.randn_like(x)

        if atom_mask != None:

            if indices == None:

                z_mean = torch.sum(z * atom_mask[:, None, :, :],
                                   dim=2)[:, :, None, :] / torch.sum(atom_mask, dim=1)[:, None, None, :]
                z = z - z_mean
            else:

                z_mean = scatter(src=z, dim=2,
                                 index=indices.unsqueeze(1),
                                 reduce="sum")
                z_mean = z_mean / (scatter(src=atom_mask[:, None, :, :], dim=2,
                                           index=indices.unsqueeze(1), reduce="sum") + 1e-8)
                z_mean = torch.stack([z_mean[i][:, indices[i], :] \
                                      for i in range(len(indices))])

                z = z - z_mean
        drift, diffusion = self.rsde.sde(x, t, means)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, atom_mask=None, indices=None, means=None):
        f, G = self.rsde.discretize(x, t, means)
        z = torch.randn_like(x)
        if atom_mask != None:

            if indices == None:

                z_mean = torch.sum(z * atom_mask[:, None, :, :],
                                   dim=2)[:, :, None, :] / torch.sum(atom_mask, dim=1)[:, None, None, :]

                z = z - z_mean

            else:

                z_mean = scatter(src=z, dim=2,
                                 index=indices.unsqueeze(1),
                                 reduce="sum")
                z_mean = z_mean / (scatter(src=atom_mask[:, None, :, :], dim=2,
                                           index=indices.unsqueeze(1), reduce="sum") + 1e-8)
                z_mean = torch.stack([z_mean[i][:, indices[i], :] \
                                      for i in range(len(indices))])

                z = z - z_mean

        x_mean = x - f

        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE) \
                and not isinstance(sde, sde_lib.cmVPSDE) \
                and not isinstance(sde, sde_lib.cmsubVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t, atom_mask=None, indices=None):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)

            if atom_mask != None:

                if indices == None:

                    noise_mean = torch.sum(noise * atom_mask[:, None, :, :],
                                           dim=2)[:, :, None, :] / torch.sum(atom_mask, dim=1)[:, None, None, :]

                    noise = noise - noise_mean
                else:

                    noise_mean = scatter(src=noise, dim=2,
                                         index=indices.unsqueeze(1),
                                         reduce="sum")
                    noise_mean = noise_mean / (scatter(src=atom_mask[:, None, :, :], dim=2,
                                                       index=indices.unsqueeze(1), reduce="sum") + 1e-8)
                    noise_mean = torch.stack([noise_mean[i][:, indices[i], :] \
                                              for i in range(len(indices))])

                    noise = noise - noise_mean
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


def shared_predictor_update_fn(x, t, sde, score_fn, predictor, probability_flow, continuous,
                               atom_mask=None, indices=None, means=None):
    """A wrapper that configures and returns the update function of predictors."""

    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, atom_mask, indices, means)


def shared_corrector_update_fn(x, t, sde, score_fn, corrector, continuous, snr, n_steps, atom_mask=None, indices=None):
    """A wrapper tha configures and returns the update function of correctors."""
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t, atom_mask, indices)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', atom_mask=None, indices=None,
                   means=None):
    """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous,
                                            atom_mask=atom_mask,
                                            indices=indices,
                                            means=means)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps,
                                            atom_mask=atom_mask,
                                            indices=indices)

    def pc_sampler(model):
        """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            if not isinstance(sde, sde_lib.cmVPSDE) and not isinstance(sde, sde_lib.cmsubVPSDE):
                x = sde.prior_sampling(shape).to(device)
            else:
                x = sde.prior_sampling(shape, atom_mask, indices=indices, means=means).to(device)

            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]

                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, score_fn=model)
                x, x_mean = predictor_update_fn(x, vec_t, score_fn=model)
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

    return pc_sampler


class TimeOutException(Exception):
    pass


num_eval = 0


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK23', eps=1e-3, device='cuda', atom_mask=None, indices=None,
                    means=None, multiple_points=False, limit=2000):
    """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

    def denoise_update_fn(score_fn, x):
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps, means=means)
        return x

    def drift_fn(score_fn, x, t, means):
        """Get the drift function of the reverse-time SDE."""
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, means)[0]

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """

        with torch.no_grad():
            global num_eval
            num_eval = 0
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                if not isinstance(sde, sde_lib.cmVPSDE) and not isinstance(sde, sde_lib.cmsubVPSDE):
                    x = sde.prior_sampling(shape).to(device)
                else:
                    x, _ = sde.prior_sampling(shape, atom_mask, indices=indices, means=means)
                    x.to(device)
            else:
                x = z

            def ode_func(t, x):

                global num_eval
                num_eval += 1
                if num_eval > limit:
                    raise TimeOutException("Exceeded the maximum number of evals")

                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t, means) * atom_mask[:, None, :, :]
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            if not multiple_points:
                solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                               rtol=rtol, atol=atol, method=method)
            else:

                solution = integrate.solve_ivp(ode_func, (sde.T, eps),
                                               to_flattened_numpy(x),
                                               rtol=rtol, atol=atol, method=method,
                                               t_eval=[sde.T - (sde.T - eps) * i / 50 for i in range(50)])
            nfe = solution.nfev

            if not multiple_points:

                x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
            else:
                shape_ = (shape[0], shape[2], shape[3], 50)
                x = torch.tensor(solution.y).reshape(shape_).to(device).type(torch.float32).transpose(-1, -2).transpose(
                    -2, -3)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler


def get_differentiable_ode_sampler(sde, shape, rtol=1e-5, atol=1e-5,
                                   eps=1e-3, device='cuda', atom_mask=None, indices=None,
                                   means=None, likelihood=False, num_centers=None, only_likelihood=False):
    """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.
    likelihood: whether to compute likelihood during sampling
    only_likelihood: whether to only compute likelihood on input sample

  Returns:
    A sampling function that returns samples and the likelihood of sampled/input data.
  """

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`, or input sample if
      only_likelihood=True.
    Returns:
      samples, number of function evaluations.
    """

        # Initial sample
        if z is None:
            # If not represent, sample the latent code from the prior distibution of the SDE.
            if not isinstance(sde, sde_lib.cmVPSDE) and not isinstance(sde, sde_lib.cmsubVPSDE):
                x = sde.prior_sampling(shape).to(device)
            else:
                x, _ = sde.prior_sampling(shape, atom_mask, indices=indices, means=means)
                x.to(device)
        else:
            x = z

        if not x.requires_grad:
            x.requires_grad = True

        if not only_likelihood:
            times = torch.tensor(
                [sde.T, eps]).to(device)
        else:

            times = torch.tensor(
                [eps, sde.T]).to(device)

        # Black-box ODE solver for the probability flow ODE
        if only_likelihood:

            init_logp = torch.zeros((x.shape[0], x.shape[1])).to(device)

            init_logp.requires_grad = True

            solution, delta_logp = odeint(model, (x, init_logp),
                                          times, rtol=rtol, atol=atol, method="scipy_solver",
                                          options={"solver": "RK23"}, adjoint_method="euler",
                                          adjoint_options={"step_size": 5e-2})

            if isinstance(sde, sde_lib.cmVPSDE) or isinstance(sde, sde_lib.cmsubVPSDE):
                prior_logp = sde.prior_logp(solution[1], num_centers, atom_mask[:x.shape[0]])
            else:
                prior_logp = sde.prior_logp(solution[1], atom_mask[:x.shape[0]])

            return prior_logp + delta_logp[1]

        elif not likelihood:
            solution = odeint(model, x, times, rtol=rtol, atol=atol, method="scipy_solver", options={"solver": "RK23"})
            return solution[1]
        else:
            init_logp = torch.zeros((x.shape[0], x.shape[1])).to(device)

            init_logp.requires_grad = True

            solution, delta_logp = odeint(model, (x, init_logp),
                                          times, rtol=rtol, atol=atol, method="scipy_solver",
                                          options={"solver": "RK23"}, adjoint_method="euler",
                                          adjoint_options={"step_size": 5e-2})

            if isinstance(sde, sde_lib.cmVPSDE) or isinstance(sde, sde_lib.cmsubVPSDE):
                prior_logp = sde.prior_logp(x, num_centers, atom_mask[:x.shape[0]])
            else:
                prior_logp = sde.prior_logp(x, atom_mask[:x.shape[0]])

            return solution[1], delta_logp[1], prior_logp

    return ode_sampler


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps, mask, exact_trace=False):

        with torch.enable_grad():
            x.requires_grad_(True)

            if exact_trace:
                trace_mask, indices = get_trace_computation_tensors(x)
                initial_size = x.shape[1]
                x = x.repeat(1, x.shape[-1] * x.shape[-2], 1, 1)

                fn_output = fn(x, t) * mask[:, None, :, :]

                fn_output_original = fn_output[:, :initial_size].clone()

                fn_output = fn_output * trace_mask

                jacobian_trace = torch.autograd.grad(fn_output, x, trace_mask)[0] * trace_mask

                jacobian_trace = torch.sum(scatter_add(src=jacobian_trace, dim=1, index=indices), dim=(-1, -2))

            else:

                fn_output = fn(x, t) * mask[:, None, :, :]

                fn_output_original = fn_output.clone()

                grad_fn_eps = torch.autograd.grad(fn_output, x, eps)[0]

                jacobian_trace = torch.sum(grad_fn_eps * eps * mask[:, None, :, :], dim=tuple(range(2, len(x.shape))))

                x.requires_grad_(False)

        return jacobian_trace, fn_output_original

    return div_fn


num_eval = 0


def get_sampling_likelihood_fn(sde, shape, inverse_scaler, device='cuda', hutchinson_type='Rademacher',
                               rtol=1e-5, atol=1e-5, method='RK23', eps=1e-5, atom_mask=None, indices=None,
                               means=None, num_centers=None, exact_trace=False, limit=2000):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

    def drift_fn(score_fn, x, t, means):
        """The drift function of the reverse-time SDE."""

        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, means)[0]

    def div_fn(model, x, t, noise, means, mask, exact_trace):
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt, means))(x, t,
                                                                         noise, mask,
                                                                         exact_trace)

    def likelihood_fn(model, data=None, reverse=True):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A score model.
      data: A PyTorch tensor.

    Returns:
      logp: A PyTorch tensor of shape [batch size]. The log-likelihoods on sampled/input data
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
        with torch.no_grad():
            global num_eval
            num_eval = 0

            if data is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                if not isinstance(sde, sde_lib.cmVPSDE) and not isinstance(sde, sde_lib.cmsubVPSDE):
                    data = sde.prior_sampling(shape).to(device)
                else:
                    data, data_init = sde.prior_sampling(shape, atom_mask, indices=indices, means=means)

                    data = data.to(device)

            else:

                if isinstance(sde, sde_lib.cmVPSDE) or isinstance(sde, sde_lib.cmsubVPSDE):
                    data_init = (data - means) * atom_mask[:, None, :, :]

            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                global num_eval
                num_eval += 1
                if num_eval > limit:
                    raise TimeOutException("Exceeded the maximum number of evals")

                sample = from_flattened_numpy(x[:-shape[0] * shape[1]], shape).to(data.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                logp_grad, drift = div_fn(model, sample, vec_t, epsilon, means, atom_mask,
                                          exact_trace=exact_trace)
                logp_grad = to_flattened_numpy(logp_grad)
                drift = to_flattened_numpy(drift)
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0] * shape[1],))], axis=0)
            if not reverse:
                solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
            else:
                solution = integrate.solve_ivp(ode_func, (sde.T, eps), init, rtol=rtol, atol=atol, method=method)

            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = from_flattened_numpy(zp[:-shape[0] * shape[1]], shape).to(data.device).type(torch.float32)
            delta_logp = from_flattened_numpy(zp[-shape[0] * shape[1]:], (shape[0], shape[1],)).to(data.device).type(
                torch.float32)

            if not reverse:
                if isinstance(sde, sde_lib.cmVPSDE) or isinstance(sde, sde_lib.cmsubVPSDE):
                    prior_logp = sde.prior_logp(z, num_centers, atom_mask[:z.shape[0]])
                else:
                    prior_logp = sde.prior_logp(z, atom_mask[:z.shape[0]])
            else:

                if isinstance(sde, sde_lib.cmVPSDE) or isinstance(sde, sde_lib.cmsubVPSDE):
                    prior_logp = sde.prior_logp(data_init, num_centers, atom_mask[:z.shape[0]])
                else:
                    prior_logp = sde.prior_logp(data, atom_mask[:z.shape[0]])

            if not reverse:
                logp = prior_logp + delta_logp
            else:
                logp = prior_logp - delta_logp

        return logp, z, nfe, delta_logp, prior_logp

    return likelihood_fn
