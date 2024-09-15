from abc import ABC, abstractclassmethod
import math
import torch
from torch import Tensor
from typing import List, Literal, Optional, TypeAlias

from diffunc.guided_diffusion import GaussianDiffusion, batched_times
from diffunc.unit_distributions import ExponentialDistribution
from diffunc.softrect import (
    softrect_inlier,
    log_softrect_inlier,
    log_softrect_outlier
)
from diffunc.sam import Segmentation

# =============================================================================
# region Class ScheduledParams
# =============================================================================

Params: TypeAlias = dict[str, float]

class ScheduledParams (dict):
    def get_params(self, t: int, num_timesteps: int) -> Params:
        """
        Args:
            t (int): the timestep in [-1, T-2]
            num_timesteps (int): the total number of timesteps T
        """
        assert -1 <= t <= num_timesteps - 2

        # let p denote the progress in the reverse diffusion from 0 (at t=T-2)
        # to 1 (at T=-1)
        p = 1 - (t + 1) / (num_timesteps - 1)

        params = {}
        for k, (vl, vr, schedule) in self.items():
            vl: float
            vr: float
            schedule: Literal['fixed', 'linear', 'log']

            # return the right parameter
            if schedule == 'fixed':
                v = vr
            # interpolate between endpoints linearly
            elif schedule == 'linear':
                v = (1 - p) * vl + p * vr
            # interpolate between endpoints on a log scale
            # increase is slow in the beginning but fast later
            elif schedule == 'log':
                log_v = (1 - p) * math.log(vl) + p * math.log(vr)
                v = math.exp(log_v)
            else:
                raise ValueError(f"Invalid schedule type: {schedule} for param {k}")

            params[k] = v

        return params

# endregion
# =============================================================================
# region Objectives
# =============================================================================

class Objective (ABC):
    def __init__(self, guidance_scale):
        self.guidance_scale = guidance_scale

    @abstractclassmethod
    def log_r(self, x: Tensor, x_targ: Tensor, params: Params) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image to condition on.
            params (Params): the parameters of the objective function.

        Returns:
            Tensor, dtype=float, shape=(B,): corresponds to conditional
                log-PDF of the objective.
        """
        return NotImplementedError

    def scaled_log_r(self, x: Tensor, x_targ: Tensor, params: Params) -> Tensor:
        return self.guidance_scale * self.log_r(x, x_targ, params)

class L2Objective (Objective):
    def log_r(self, x: Tensor, x_targ: Tensor, params: Params) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image to condition on.
            params (Params): the parameters of the objective function.

        Returns:
            Tensor, dtype=float, shape=(B,): the mean pixel-wise L2 distance.
        """
        return -(x - x_targ).square().mean(dim=(1, 2, 3))  # (B,)

class L1Objective (Objective):
    def log_r(self, x: Tensor, x_targ: Tensor, params: Params) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image to condition on.
            params (Params): the parameters of the objective function.

        Returns:
            Tensor, dtype=float, shape=(B,): the mean pixel-wise L1 distance.
        """
        return -(x - x_targ).abs().mean(dim=(1, 2, 3))  # (B,)

class ExponentialPriorSPMObjective (Objective):
    def log_r(self, x: Tensor, x_targ: Tensor, params: Params) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image to condition on.
            params (Params): the parameters of the objective function.

        Returns:
            Tensor, dtype=float, shape=(B,): the unnormalized log-PDF of the
                soft pixel inlier fraction under the exponential distribution
                parametrized by λ.
        """
        # get parameters
        width = params['width']
        steepness = params['steepness']
        λ = params['λ']

        # compute the inlier score. an inlier is where *all* R/G/B channels are inliers.
        inlier_score = softrect_inlier(x, center=x_targ, width=width, steepness=steepness)  # (B, 3, H, W)
        inlier_score = inlier_score.prod(dim=1)  # (B, H, W)  multiply across RGB channels

        # check that inlier score lies in [0, 1]
        assert not inlier_score.isnan().any()  # no NaNs
        assert not inlier_score.isinf().any()  # no Infs
        assert (0.0 <= inlier_score).all() and (inlier_score <= 1.0).all()  # lies in [0, 1]

        outlier_score = 1 - inlier_score  # (B, H, W)
        outlier_fraction = outlier_score.mean(dim=(1, 2))  # (B,)
        return ExponentialDistribution(λ).log_pdf(outlier_fraction)  # (B,)

class MRFPriorSPMObjective (Objective):
    def log_r(self, x: Tensor, x_targ: Tensor, params: Params) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image to condition on.
            params (Params): the parameters of the objective function.

        Returns:
            Tensor, dtype=float, shape=(B,): the unnormalized log-PDF of the
                soft pixel inlier fraction under the MRF prior.
        """
        # get parameters
        width = params['width']
        steepness = params['steepness']

        # compute pixel-wise outliers
        inlier = softrect_inlier(x, center=x_targ, width=width, steepness=steepness)  # (B, 3, H, W)
        inlier = inlier.prod(dim=1)  # (B, H, W)  equivalent to multiplying softrect across RGB channels
        
        # TODO (Sid): try a different definition of soft outliers using the
        # non-log version of `softrect.log_softrect_outlier()` function.
        # compute the outlier score.
        outlier = 1 - inlier  # (B, H, W)

        # node factors
        node_factor = inlier  # (B,)

        # left neighbors
        l_inlier  =  inlier[:, :, :-1]  # (B, H, W-1)
        l_outlier = outlier[:, :, :-1]  # (B, H, W-1)

        # right neighbors
        r_inlier  =  inlier[:, :, +1:]  # (B, H, W-1)
        r_outlier = outlier[:, :, +1:]  # (B, H, W-1)

        # top neighbors
        t_inlier  =  inlier[:, :-1, :]  # (B, H-1, W)
        t_outlier = outlier[:, :-1, :]  # (B, H-1, W)

        # bottom neighbors
        b_inlier  =  inlier[:, +1:, :]  # (B, H-1, W)
        b_outlier = outlier[:, +1:, :]  # (B, H-1, W)

        # edge factors: either both vertices are inliers or both are outliers.
        h_edge_factor = (l_inlier * r_inlier) + (l_outlier * r_outlier)  # (B, H, W-1)
        v_edge_factor = (t_inlier * b_inlier) + (t_outlier * b_outlier)  # (B, H-1, W)

        # take mean of log-factors across pixels
        node_factor   =   node_factor.mean(dim=(1, 2))  # (B,)
        h_edge_factor = h_edge_factor.mean(dim=(1, 2))  # (B,)
        v_edge_factor = v_edge_factor.mean(dim=(1, 2))  # (B,)

        assert node_factor.isfinite().all()
        assert h_edge_factor.isfinite().all()
        assert v_edge_factor.isfinite().all()

        # return node_factor  # (B,)
        return node_factor + 0.47 * h_edge_factor + 0.47 * v_edge_factor  # (B,)

class SAMPriorSPMObjective (Objective):
    def __init__(self, seg: Segmentation, guidance_scale: float):
        """
        Args:
            segs (Segmentation): N disjoint segmentation masks of size (H, W).
        """
        super().__init__(guidance_scale)
        self.seg = seg

    def log_r(self, x: Tensor, x_targ: Tensor, params: Params) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image to condition on.
            params (Params): the parameters of the objective function.

        Returns:
            Tensor, dtype=float, shape=(B,): the unnormalized log-PDF of the
                segment-anything prior. All pixels in a segment are conditioned
                to be either all-inliers or all-outliers.
        """
        # get parameters
        width = params['width']
        steepness = params['steepness']

        # compute pixel-wise inliers: an inlier is where *all* R/G/B channels are inliers.
        log_inlier = log_softrect_inlier(x, center=x_targ, width=width, steepness=steepness)  # (B, 3, H, W)
        log_inlier = log_inlier.sum(dim=1)  # (B, H, W) equivalent to AND across RGB channels
        
        # compute pixel-wise outliers: an outlier is where *any* R/G/B channel is an outlier.
        log_outlier = log_softrect_outlier(x, center=x_targ, width=width, steepness=steepness)  # (B, 3, H, W)
        log_outlier = log_outlier.logsumexp(dim=1)  # (B, H, W) equivalent to OR across RGB channels

        # log-score of all pixels inside segment being inliers
        log_all_inlier = log_inlier.unsqueeze(dim=1)  # (B, 1, H, W)
        log_all_inlier = (self.seg.masks * log_all_inlier).mean(dim=(2, 3))  # (B, N) equivalent to AND across all seg pixels

        # log-score of all pixels inside segment being outliers
        log_all_outlier = log_outlier.unsqueeze(dim=1)  # (B, 1, H, W)
        log_all_outlier = (self.seg.masks * log_all_outlier).mean(dim=(2, 3))  # (B, N) equivalent to AND across all seg pixels

        # log-score of all pixels inside segment being all-inliers or all-outliers
        log_all_same = torch.logaddexp(log_all_inlier, log_all_outlier)  # (B, N) equivalent to OR

        # return average across segments
        return log_all_same.mean(dim=1)  # (B,)

# endregion
# =============================================================================
# region Abstract class Conditioner
# =============================================================================

class Conditioner:
    def __init__(self,
                 diffusion: GaussianDiffusion,
                 objectives: List[Objective],
                 params: Optional[ScheduledParams] = None,
                 apply_q_transform: bool=False,
                 apply_p_transform: bool=False):
        """
        Args:
            diffusion (GaussianDiffusion): the main GaussianDiffusion instance.
            objective (List[Objective]): a list of objective functions used for
                conditioning.
            params (ScheduledParams, optional): the parameters of the objective
                function that potentially depends on the timestep.
            apply_q_transform (bool): whether to scale the target image x_targ
                to timestep t=t as defined in q_sample.
                This computes x'_targ ← 𝔼 [xₜ | x_start=x_targ ] before computing
                log_r(x, x'_targ, t).
            apply_p_transform (bool): whether to scale the predicted image xₜ
                to timestep t=start using the learned diffusion model 𝛍ᶿ(⋅).
                This computes x' ← 𝔼 [x_start | xₜ=x, θ] before computing
                log_r(x', x_targ, t).
        """
        self.diffusion = diffusion
        self.objectives = objectives
        self.params = params
        self.apply_q_transform = apply_q_transform
        self.apply_p_transform = apply_p_transform

        assert not (apply_q_transform and apply_p_transform), \
            "Cannot apply both q and p transforms"

    @property
    def num_timesteps(self) -> int:
        return self.diffusion.num_timesteps

    def scaled_log_r(self, x: Tensor, t: int, x_targ: Tensor,) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            t (int): the timestep in [-1, T-2]
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image to condition on.

        Returns:
            Tensor, dtype=float, shape=(B,): the log-PDF of the soft pixel
                match fraction under the exponential distribution.
        """
        if self.params is not None:
            params = self.params.get_params(t, self.num_timesteps) # (B,)
        else:
            params = {}  # empty

        # add log_r's across all objectives
        scaled_log_r = 0
        for obj in self.objectives:
            scaled_log_r += obj.scaled_log_r(x, x_targ, params)  # (B,)
        return scaled_log_r  # (B,)

    def q_transform(self, x_targ: Tensor, t: int) -> Tensor:
        """
        Args:
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
                at the starting timestep i.e. t == 'start' == -1.
            t (int): the timestep in [-1, T-2] to transform x_targ to.
        Returns:
            (Tensor, dtype=float, shape=(B, 3, H, W)): the normalized target
                image x_targ at t=start scaled to timestep t under the forward
                diffusion process (q_sample) which is 𝔼 [xₜ | x_start=x_targ ].
        """
        if t == -1:
            return x_targ  # (B, 3, H, W)
        else:
            # return self.sqrt_alphas_cumprod[t] * x_targ
            t = batched_times(t, x_targ)  # (B,)
            return self.diffusion.q_sample(x_targ, t, noise=0)  # (B, 3, H, W)

    def p_transform(self, x: Tensor, t: int) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
                at timestep t.
            t (int): the timestep in [-1, T-2].
        Returns:
            (Tensor, dtype=float, shape=(B, 3, H, W)): x at timestep t
                transformed to timestep t == 'start' == -1 under the reverse
                diffusion process using the trained diffusion model:
                𝔼 [x_start | xₜ=x, θ].
        """
        if t == -1:
            return x  # (B, 3, H, W)
        else:
            assert self.diffusion.self_condition is False, \
                "Self-conditioned diffusion currently not supported by p_transform"
            t = batched_times(t, x)  # (B,)
            preds = self.diffusion.model_predictions(x, t, x_self_cond=False, clip_x_start=False)  # (B, 3, H, W)
            return preds.pred_x_start  # (B, 3, H, W)

    @torch.enable_grad()  # important since we use torch.autograd.grad!
    def __call__(self, x: Tensor, t: int, x_targ: Tensor) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            t (int): the timestep in [-1, T-2]
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image to condition on.
        """
        # check that t lies in [-1, T-2]
        assert -1 <= t and t <= self.num_timesteps - 2

        x_in = x.detach().requires_grad_(True)  # (B, 3, H, W)

        if self.apply_q_transform:
            x_targ = self.q_transform(x_targ, t)  # (B, 3, H, W)

        if self.apply_p_transform:
            x_in = self.p_transform(x_in, t)  # (B, 3, H, W)

        log_r: Tensor = self.scaled_log_r(x_in, t, x_targ)  # (B,)
        # TODO: check if we can set is_grads_batched=True
        grad = torch.autograd.grad(log_r.sum(), x_in)[0]  # (B, 3, H, W)
        return grad  # (B, 3, H, W)

class InterpolatedConditioner (Conditioner):
    """
    Given two conditioners representing r₁(x₀) and r₂(x₀), the interpolated
    conditioner represents
                    r(x₀) = r₁(x₀)ᵗ·r₂(x₀)¹⁻ᵗ
    where t goes from (T-1) to 0 during the reverse diffusion process.
    
    In the beginning of the reverse diffusion process, the interpolated
    conditioner is equal to r₁(x₀), and at the end, it is equal to r₂(x₀).
    """
    def __init__(self, cond1: Conditioner, cond2: Conditioner):
        assert cond1.diffusion is cond2.diffusion
        assert cond1.apply_q_transform == cond2.apply_q_transform
        assert cond1.apply_p_transform == cond2.apply_p_transform
        
        # guidance scale is controlled individually by each conditioner, so
        # set guidance of the wrapper to 1
        super().__init__(diffusion=cond1.diffusion,
                         apply_q_transform=cond1.apply_q_transform,
                         apply_p_transform=cond1.apply_p_transform,
                         guidance_scale=1)

        self.cond1 = cond1
        self.cond2 = cond2

    def log_r(self, x: Tensor, t: int, x_targ: Tensor) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            t (int): the timestep in [-1, T-2]
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image to condition on.

        Returns:
            Tensor, dtype=float, shape=(B,): the log-PDF of the soft pixel
                match fraction under the exponential distribution.
        """
        # linear interpolation of gradients between the two conditioners
        α = (t + 1) / (self.num_timesteps - 1)
        scaled_log_r1 = self.cond1.scaled_log_r(x, t, x_targ)
        scaled_log_r2 = self.cond2.scaled_log_r(x, t, x_targ)
        return α * scaled_log_r1 + (1 - α) * scaled_log_r2

# endregion
# =============================================================================
