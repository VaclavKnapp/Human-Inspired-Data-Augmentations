# Sample run on 2 GPUs
# Specify the dataset in train_contrastive.yaml

CUDA_VISIBLE_DEVICES=5 torchrun \
  --nnodes=1 --nproc_per_node=1 \
  --master_port=12395 \
  -m train \
  --config-name train_contrastive \
  backbone=dinov2_large_reg \
  batch_size=32 \
  epochs=100 \
  optimizer.lr=2e-6 \
  num_workers=7 \
  use_lora=True \
  lora_r=16