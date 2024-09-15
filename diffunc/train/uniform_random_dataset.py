#!/usr/bin/env python

# copied from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/a3d0922fb8713cae3caa3a705e52b706c3a97721/README.md?plain=1#L57-L86

# to launch this training job on SuperCloud:
# LLsub diffunc/train/uniform_random_dataset.py -g volta:1

# to launch this training job on Satori:
# srun \
#     --job-name=train-ddpm \
#     --gres=gpu:1 \
#     --ntasks=1 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=5 \
#     --mem=1T \
#     --time=1440  `# the maximum time allowed on Satori (24 hours)` \
#     --kill-on-bad-exit=1 \
#     python tools/train_ddpm/rugd_trail_15.py &> srun_train_ddpm_rugd_trail_15.out&

import argparse

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from diffunc.train.uniform_random_trainer import UniformRandomImageTrainer

parser = argparse.ArgumentParser(description='Train a DDPM model on uniformly random images')
parser.add_argument('--ckpt-file', type=str, default=None, help='path to checkpoint')
args = parser.parse_args()

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = (32, 32),    # size of image (height, width) originally this was (128, 128)
    timesteps = 1000,         # number of steps
    sampling_timesteps = 250  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = UniformRandomImageTrainer(
    diffusion,
    train_batch_size = 16,  # was 32
    train_lr = 8e-5,
    train_num_steps = 10000,  # was 700000, # total training steps
    gradient_accumulate_every = 2,          # gradient accumulation steps
    ema_decay = 0.995,                      # exponential moving average decay
    amp = False,  # was True                # turn on mixed precision
    calculate_fid = False,  # was True      # whether to calculate fid during training,
    results_folder = 'results/uniform_random_dataset',
    save_and_sample_every = 1000
)

trainer.train()
