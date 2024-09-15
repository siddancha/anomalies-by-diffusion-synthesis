# from diffusers import AutoencoderKL
# from diffusers.models.modeling_outputs import AutoencoderKLOutput

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)

import torch
from torch import Tensor

class VanillaEncoderDecoder:
    def encode(self, i):
        """
        Args:
            i (Tensor, dtype=float, shape=(1, 3, H, W)): input image in RGB
                space with values in [0, 1]

        Returns:
            x (Tensor, dtype=float, shape=(1, 3, H, W)): encoded image in
                latent space with values in [-1, 1].
        """
        x = normalize_to_neg_one_to_one(i)  # (1, 3, H, W)
        return x  # (1, 3, H, W)

    def decode(self, x: Tensor):
        """
        Args:
            x (Tensor, dtype=float, shape=(1, 3, H, W)): encoded image in
                latent space with values in [-1, 1].

        Returns:
            i (Tensor, dtype=float, shape=(1, 3, H, W)): decoded image in RGB
                space with values in [0, 1].
        """
        i = unnormalize_to_zero_to_one(x)  # (1, 3, H, W)
        return i  # (1, 3, H, W)

class VaeEncodeDecoder (VanillaEncoderDecoder):
    """
    https://towardsdatascience.com/stable-diffusion-using-hugging-face-501d8dbdd8#2049
    """
    def __init__(self):
        # this gets saved to
        # ~/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            torch_dtype=torch.float16
        ).to("cuda")

    @torch.no_grad()
    def encode(self, i: Tensor):
        """
        Args:
            i (Tensor, dtype=float, shape=(1, 3, H, W)): input image in RGB
                space with values in [0, 1]

        Returns:
            x (Tensor, dtype=float, shape=(1, 4, H/8, W/8)): encoded image in
                latent space with values in [-1, 1].
        """
        i = super().encode(i)  # (1, 3, H, W)

        i = i.half()  # (1, 3, H, W)
        d: AutoencoderKLOutput = self.vae.encode(i)  # BS=(1) C=4 H=H/8 W=W/8
        x: Tensor = d.latent_dist.sample() * 0.18215  # (1, 4, H/8, W/8)
        x = x.float()  # (1, 4, H/8, W/8)

        return x
    
    @torch.no_grad()
    def decode(self, x: Tensor):
        """
        Args:
            x (Tensor, dtype=float, shape=(1, 4, H/8, W/8)): encoded image in
                latent space with values in [-1, 1].

        Returns:
            i (Tensor, dtype=float, shape=(1, 3, H, W)): decoded image in RGB
                space with values in [0, 1].
        """
        x = x.half()  # (1, 4, H/8, W/8)
        x = (1 / 0.18215) * x  # (1, 4, H/8, W/8)
        i: Tensor = self.vae.decode(x).sample  # (1, 3, H, W)
        i = i.float()  # (1, 3, H, W)

        i = super().decode(i)  # (1, 3, H, W)

        return i
