import torch
from torch import Tensor
from tqdm.auto import tqdm
from typing import Callable, Optional, Tuple, Union
from .anneal_samplers import Sampler
from .constraints import Constraint

from denoising_diffusion_pytorch.guided_diffusion import (
    GaussianDiffusion as BaseGaussianDiffusion,
    normalize_to_neg_one_to_one,
    exists,
)

# from .ed import VanillaEncoderDecoder, VaeEncodeDecoder
from .ed import VanillaEncoderDecoder

"""
Timestep notation
=================
In the DDPM paper, timesteps are numbered from 0, 1, ..., T, where T = 1000.
In the code, timesteps map as follows:
     Paper  Code
     -----  -----
       0 -> start
       1 -> 0
        ...
    1000 -> 999
Therefore, beta_999 in code corresponds to the last beta_1000 in the paper.
Note that in the paper, betas are only defined for t = 1, ..., T.

Notation for normalized and unnormalized images
===============================================
To the extent possible, I will try to refer to encoded images (lying in
[-1, 1]) by "x" and, input/decoded images (lying in [0, 1]) "img".
"""

GradientFn = Callable[[Tensor, Tensor], Tensor]

def batched_times(t: int, x: Tensor) -> Tensor:
    B, *_, device = *x.shape, x.device
    return torch.full((B,), t, device=device, dtype=torch.long)  # (B,)

class GaussianDiffusion (BaseGaussianDiffusion):
    def __init__(
        self,
        model,
        *,
        image_size: Union[int, Tuple[int, int]],  # (height, width)
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        use_vae: bool = False,  # new parameter
    ):
        super().__init__(model,
                         image_size=image_size,
                         timesteps=timesteps,
                         sampling_timesteps=sampling_timesteps,
                         objective=objective,
                         beta_schedule=beta_schedule,
                         schedule_fn_kwargs=schedule_fn_kwargs,
                         ddim_sampling_eta=ddim_sampling_eta,
                         auto_normalize=auto_normalize,
                         min_snr_loss_weight=min_snr_loss_weight,
                         min_snr_gamma=min_snr_gamma)

        if use_vae:
            self.ed = VaeEncodeDecoder()
            assert self.image_size[0] % 8 == 0
            assert self.image_size[1] % 8 == 0
            self.latent_size = (self.image_size[0] // 8, self.image_size[1] // 8)
        else:
            self.ed = VanillaEncoderDecoder()
            self.latent_size = self.image_size

    def encode(self, img: Tensor) -> Tensor:
        return self.ed.encode(img)

    def decode(self, x: Tensor) -> Tensor:
        return self.ed.decode(x)

    @property
    def T(self):
        return self.num_timesteps

    @property
    def device(self):
        return self.betas.device

    # replace self.normalize() with self.encode()
    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        x = self.encode(img)
        return self.p_losses(x, t, *args, **kwargs)

    def condition_mean(self,
                       cond_fn: GradientFn,
                       μ_curr: Tensor,
                       Σ_curr: Tensor,
                       x_next: Tensor,
                       t_next: int,
                       guidance_kwargs: Optional[dict] = None,
        ):
        """
        Compute the mean for the current step (t-1), given a function cond_fn
        that computes the gradient of a conditional log probability with
        respect to x. In particular, cond_fn computes grad(log(p(y|x))), and we
        want to condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).

        Args:
            cond_fn (GradientFn) A function that computes the gradient
                ∇xₜ log p(y | xₜ, t) with respect to xₜ, at time t in [-1, T-2].
            μ_curr (Tensor, dtype=float, shape=(B, 3, H, W)): The mean
                predicted for the current timestep t-1.
            Σ_curr (Tensor, dtype=float, shape=(B, 1, 1, 1)): The variance
                predicted for the current timestep t-1.
            x_next (Tensor, dtype=float, shape=(B, 3, H, @)): The input to the
                model in the next timestep t.
            t_next (int): time of the next timestep t, lying in [0, T-1].
            guidance_kwargs (dict, optional): Keyword arguments to pass to cond_fn.
        """
        # since we want to compute the gradient with respect to xₜ₋₁, we must
        # reduce t to t-1.
        t_curr = t_next - 1  # Now, t_curr ∈ [-1, T-2]

        # this fixes a bug in lucidrains' repo -- we use the predicted mean μ_curr
        # for the current timestep rather than the input x_next of the next timestep.
        # see https://github.com/lucidrains/denoising-diffusion-pytorch/issues/276
        gradient = cond_fn(μ_curr, t_curr, **guidance_kwargs)
        new_μ = μ_curr.float() + Σ_curr * gradient.float()
        # print("gradient: ",(variance * gradient.float()).mean())
        return new_μ

    def score(self,
              x: Tensor,
              t: int,
              cond_fn: Optional[GradientFn] = None,
              guidance_kwargs: Optional[dict] = None) -> Tensor:
        """
        The score function learned by the diffusion process.
        This is a rescaling of the learned noise function.

        Args:
            x (Tensor, dtype=float, shape=(B, 3, H, W)): **normalized** image.
            t (int): timestep, lying in [0, T-2].
            cond_fn (GradientFn) A function that computes the gradient
                ∇xₜ log p(y | xₜ, t) with respect to xₜ, at time t in [0, T-2].
            guidance_kwargs (dict, optional): Keyword arguments to pass to cond_fn.
        """
        # score function is not available for
        #   t = -1 : since there is no noise at this level, and DSM can only
        #            learn the score of a noised distribution.
        #   t = T-1: since T-1 should correspond to the standard normal
        #            distribution, and we have an exact sample from that. so
        #            we shouldn't be doing MCMC inference at this level.
        assert 0 <= t <= self.T - 2, \
            f'Score function can only be computed for t ϵ [0, {self.T - 2}] but got t={t}'

        B, *_, device = *x.shape, x.device
        batched_times = torch.full((B,), t, device=device, dtype=torch.long)  # (B,)
        preds = self.model_predictions(x=x, t=batched_times, x_self_cond=None)  # BS=(B,), H=H, W=W
        ε_t = preds.pred_noise  # (B, 3, H, W)

        # formula to convert eps to score: https://bit.ly/ddpm-score-from-eps
        # https://github.com/yilundu/reduce_reuse_recycle/blob/e5e08718fea63caeebcc1bb465b6b980f6f59283/inf_sample.py#L193-L207
        scalar = 1.0 / self.sqrt_one_minus_alphas_cumprod[t]  # (,)
        s_t = -ε_t * scalar  # (B, 3, H, W)

        if cond_fn is not None:
            # add cond_fn == ∇xₜ log p(y | xₜ, t)
            gradient = cond_fn(x, t, **guidance_kwargs)  # (B, 3, H, W)
            s_t = s_t + gradient  # (B, 3, H, W)

        return s_t

    @torch.no_grad()
    def p_sample(self,
                 x: Tensor,
                 t: int,
                 x_self_cond = None,
                 cond_fn: Optional[Callable] = None,
                 guidance_kwargs: Optional[dict] = None
        ):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True
        )
        if exists(cond_fn) and exists(guidance_kwargs):
            model_mean = self.condition_mean(cond_fn, model_mean, variance, x, t, guidance_kwargs)
        
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop_from_intermediate_level(
            self,
            x_begin: Tensor,
            t_begin: int,
            cond_fn: Optional[GradientFn] = None,
            sampler: Optional[Sampler] = None,
            constraint: Optional[Constraint] = None,
            guidance_kwargs: Optional[dict] = None,
        ):
        """
        Based on super().p_sample_loop() but starting from intermediate timestep.

        Args:
            x_begin (Tensor, dtype=float, shape=(B, 3, H, W)): **normalized**
                image to start the reverse diffusion process from.
            t_begin (int): timestep to start the reverse diffusion process
                from, lying between 0 and T-1.
            cond_fn (GradientFn, optional): A function that computes the gradient
                ∇xₜ log p(y | xₜ, t) with respect to xₜ, at time t in [-1, T-2].
            guidance_kwargs (dict, optional): Keyword arguments to pass to cond_fn.
        """
        x = x_begin  # (B, 3, H, W)
        yield x

        x_start = None

        for t in tqdm(reversed(range(0, t_begin+1)), desc = 'sampling loop time step', total = t_begin+1):
            self_cond = x_start if self.self_condition else None
            x, x_start = self.p_sample(x, t, self_cond, cond_fn, guidance_kwargs)
            t = t - 1  # now we're at the previous timestep

            if (sampler is not None) and (t >= 0):
                x = sampler.sample_step(x, t, cond_fn, guidance_kwargs)

            if (constraint is not None):
                x = constraint.apply(x, t, **guidance_kwargs)

            yield x

    @torch.no_grad()
    def p_sample_loop(self,
                      batch_size: int,
                      cond_fn: Optional[GradientFn] = None,
                      sampler: Optional[Sampler] = None,
                      constraint: Optional[Constraint] = None,
                      guidance_kwargs: Optional[dict] = None,
        ):
        """
        Args:
            batch_size (int): Number of images to sample.
            cond_fn (GradientFn, optional): A function that computes the gradient
                ∇xₜ log p(y | xₜ, t) with respect to xₜ, at time t in [-1, T-2].
            guidance_kwargs (dict, optional): Keyword arguments to pass to cond_fn.
        """
        x_begin = torch.randn((batch_size, self.channels, *self.latent_size), device=self.device)
        t_begin = self.num_timesteps - 1

        yield from self.p_sample_loop_from_intermediate_level(
            x_begin, t_begin, cond_fn, sampler, constraint, guidance_kwargs
        )

    @torch.no_grad()
    def ddim_sample(self, batch_size: int):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn((batch_size, self.channels, *self.latent_size), device = device)
        yield x

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch_size,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(x, time_cond, self_cond, clip_x_start = True)

            if time_next < 0:
                x = x_start
                yield x
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            yield x

    @torch.no_grad()
    def sample(self, batch_size = 16):
        """Mainly used for training"""
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        for x in sample_fn(batch_size): pass
        return self.decode(x)

    @torch.inference_mode()
    def ae_from_intermediate_level(
            self,
            x_start: Tensor,
            t_begin: int):
        """
        Args:
            x_start (Tensor, dtype=float, shape=(1, 3, H, W)): *normalized*
                input image at t=start.
            t_begin (int): intermediate timestep to encode to and then decode
                from, lying between 0 and T-1.
        """
        t_tensor = torch.tensor([t_begin],
                                device=torch.cuda.current_device(),
                                dtype=torch.long)
        x_begin = self.q_sample(x_start, t_tensor)
        yield from self.p_sample_loop_from_intermediate_level(x_begin, t_begin)
