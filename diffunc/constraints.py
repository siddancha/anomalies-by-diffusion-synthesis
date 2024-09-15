from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .sam import Segmentation
    from .conditioner import ScheduledParams
    from .guided_diffusion import GaussianDiffusion

class Constraint (ABC):
    @abstractmethod
    def apply(self, x: Tensor, t: int, x_targ: Tensor) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            t (int): the timestep in [-1, T-2]
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image used for constraints.
        
        Returns:
            (Tensor, dtype=float, shape=(B, 3, H, W)): the image with the hard
                constraint applied.
        """
        return NotImplementedError

class SegInlierOutlierConstraint:
    def __init__(self,
                 diffusion: GaussianDiffusion,
                 seg: Segmentation,
                 params: Optional[ScheduledParams] = None):
        """
        Args:
            segs (Tensor, dtype=bool, shape=(N, H, W)): N disjoint segmentation
                masks.
            window (float): the window size in [-1, 1] for the inlier/outlier
        """
        self.diffusion = diffusion
        self.seg = seg
        self.params = params

    def apply(self, x: Tensor, t: int, x_targ: Tensor) -> Tensor:
        """
        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): normalized image
            t (int): the timestep in [-1, T-2]
            x_targ (Tensor, dtype=float, shape=(B, 3, H, W)): normalized target
                image used for constraints.
        
        Returns:
            (Tensor, dtype=float, shape=(B, 3, H, W)): the image with the hard
                constraint applied.
        """
        params = self.params.get_params(t, self.diffusion.num_timesteps)
        window = params['window']

        B, _, H, W, N = *x.shape, self.seg.count

        l_edge = (x_targ - window / 2).clamp(min=-1)  # (B, 3, H, W)
        r_edge = (x_targ + window / 2).clamp(max=+1)  # (B, 3, H, W)

        # pixel-wise inliers
        is_pix_inlier = (l_edge <= x) & (x <= r_edge)  # (B, 3, H, W)
        is_pix_inlier = is_pix_inlier.all(dim=-3, keepdim=True)  # (B, 1, H, W)

        # segment-wise inliers
        seg_inlier_count = is_pix_inlier & self.seg.masks  # (B, N, H, W)
        seg_inlier_count = seg_inlier_count.sum(dim=(-2, -1))  # (B, N)
        is_seg_inlier = seg_inlier_count >= 0.5 * self.seg.sizes  # (B, N)

        # gather seg-level status to pixel-level
        is_seg_inlier = is_seg_inlier.view(B, N, 1, 1)  # (B, N, 1, 1)
        is_seg_inlier = is_seg_inlier.expand(-1, -1, H, W)  # (B, N, H, W)
        lookup = self.seg.lookup.expand(B, 1, H, W)  # (1, 1, H, W)
        is_my_seg_inlier = is_seg_inlier.gather(dim=1, index=lookup)
        is_my_seg_inlier = is_my_seg_inlier.squeeze(dim=1)  # (B, H, W)

        # pixels that are to be made inliers
        make_seg_inlier = is_my_seg_inlier & self.seg.foreground  # (B, H, W)
        make_seg_outlier = (~is_my_seg_inlier) & self.seg.foreground  # (B, H, W)

        # forced to become inliers
        made_inliers = x.clamp(min=l_edge, max=r_edge)  # (B, 3, H, W)
        x = torch.where(make_seg_inlier, made_inliers, x)  # (B, 3, H, W)

        # forced to become outliers
        # TODO: currently, all 3 RGB channels are clamped - but to me made an
        # outlier, only one channel needs to be clamped. Clamp channels that
        # requires the minimum perturbation.
        made_outliers = torch.where(
            x < x_targ,  # (B, 3, H, W)
            x.clamp(max=l_edge),  # (B, 3, H, W)
            x.clamp(min=r_edge),  # (B, 3, H, W)
        )
        x = torch.where(make_seg_outlier, made_outliers, x)  # (B, 3, H, W)

        return x  # (B, 3, H, W)
