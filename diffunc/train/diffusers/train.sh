#!/bin/sh

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="data/cityscapes/leftImg8bit"
export OUTPUT_DIR="results/cityscapes_diffusers"

accelerate launch diffunc/train/diffusers/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=200000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=10000 \
  --output_dir=$OUTPUT_DIR
