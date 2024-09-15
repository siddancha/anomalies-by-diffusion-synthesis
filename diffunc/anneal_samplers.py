"""
Copied and modified from
https://github.com/yilundu/reduce_reuse_recycle/blob/main/anneal_samplers.py
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import Tensor
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .guided_diffusion import GaussianDiffusion, GradientFn

class Sampler (ABC):
    def __init__(self, diffusion: GaussianDiffusion):
        self._diffusion = diffusion

    @abstractmethod
    def sample_step(self,
                    x: Tensor,
                    t: int,
                    cond_fn: Optional[GradientFn] = None,
                    guidance_kwargs: Optional[dict] = None):
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): **normalized** image.
            t (int): timestep, lying in [0, T-2].
            cond_fn (GradientFn): A function that computes the gradient
                ∇xₜ log p(y | xₜ, t) with respect to xₜ, at time t in [0, T-2].
            guidance_kwargs (dict, optional): Keyword arguments to pass to cond_fn.
        """
        raise NotImplementedError

def leapfrog_step(x_0,
                  v_0,
                  gradient_target,
                  step_size,
                  mass_diag_sqrt,
                  num_steps,
                  ):
    """Multiple leapfrog steps with no metropolis correction."""
    x_k = x_0
    v_k = v_0
    if mass_diag_sqrt is None:
        mass_diag_sqrt = torch.ones_like(x_k)

    mass_diag = mass_diag_sqrt ** 2.
    grad = gradient_target(x_k)
    for _ in range(num_steps):    # Inefficient version - should combine half steps
        v_k += 0.5 * step_size * grad#gradient_target(x_k)    # half step in v
        x_k += step_size * v_k / mass_diag    # Step in x
        grad = gradient_target(x_k)
        v_k += 0.5 * step_size * grad    # half step in v
    return x_k, v_k

class AnnealedUHASampler (Sampler):
    """Implements UHA Sampling"""
    def __init__(self,
                 num_steps,
                 num_samples_per_step,
                 step_sizes,
                 damping_coeff,
                 mass_diag_sqrt,
                 num_leapfrog_steps,
                 gradient_function,
                 ):
        assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
        self._damping_coeff = damping_coeff
        self._mass_diag_sqrt = mass_diag_sqrt
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        self._num_leapfrog_steps = num_leapfrog_steps
        self._num_samples_per_step = num_samples_per_step
        self._gradient_function = gradient_function

    def leapfrog_step_(self, x, v, i, ts, model_args):
            step_size = self._step_sizes[i]
            return leapfrog_step(x, v, lambda _x: self._gradient_function(_x, ts, **model_args), step_size, self._mass_diag_sqrt[i], self._num_leapfrog_steps)

    def sample_step(self, x, t,ts, model_args):
        # Sample Momentum
        v = torch.randn_like(x) * self._mass_diag_sqrt[t]

        for i in range(self._num_samples_per_step):
            # Partial Momentum Refreshment
            eps = torch.randn_like(x)
            v = v * self._damping_coeff + np.sqrt(1. - self._damping_coeff**2) * eps * self._mass_diag_sqrt[t]
            x, v = self.leapfrog_step_(x, v, t, ts, model_args)

        return x

class AnnealedULASampler (Sampler):
    """Implements AIS with ULA"""
    def __init__(self, diffusion: GaussianDiffusion, num_samples_per_step: int):
        super().__init__(diffusion)
        self._num_samples_per_step = num_samples_per_step
        self._step_sizes: Tensor = self._diffusion.betas  # (T,)

    def sample_step(self,
                    x: Tensor,
                    t: int,
                    cond_fn: Optional[GradientFn] = None,
                    guidance_kwargs: Optional[dict] = None):
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): **normalized** image.
            t (int): timestep, lying in [0, T-2].
            cond_fn (GradientFn): A function that computes the gradient
                ∇xₜ log p(y | xₜ, t) with respect to xₜ, at time t in [0, T-2].
            guidance_kwargs (dict, optional): Keyword arguments to pass to cond_fn.
        """
        for _ in range(self._num_samples_per_step):
            ss = self._step_sizes[t]  # (,)
            std = (2 * ss).sqrt()  # (,)
            grad = self._diffusion.score(x, t, cond_fn, guidance_kwargs)  # (B, 3, H, W)
            noise = torch.randn_like(grad) * std  # (B, 3, H, W)
            x = x + grad * ss + noise  # (B, 3, H, W)
        return x
