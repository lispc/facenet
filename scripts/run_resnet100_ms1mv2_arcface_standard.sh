#!/usr/bin/env bash
# ResNet100-IR + ArcFace on MS1MV2 LMDB, aligned with standard ArcFace/InsightFace config.
# Key differences from earlier experiments:
# - input_size 112 (was 224)
# - embedding_dim 512 (was 128)
# - per-GPU P=64 K=2 -> global batch 512, no gradient accumulation
# - SGD lr=0.1, step scheduler, ~180k total steps
# - gradient checkpointing disabled
set -euo pipefail

OUTPUT_DIR="./checkpoints/resnet100_ms1mv2_lmdb_p64k2_16ep_arcface_standard"
mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node=4 train.py \
  --data_root ./data/ms1mv2.lmdb \
  --dataset_type lmdb \
  --preload_lmdb \
  --model iresnet100 \
  --input_size 112 \
  --embedding_dim 512 \
  --dropout 0.5 \
  --p 64 \
  --k 2 \
  --num_batches_per_epoch 11376 \
  --epochs 16 \
  --optimizer sgd \
  --lr 1e-1 \
  --min_lr 1e-6 \
  --weight_decay 5e-4 \
  --scheduler step \
  --warmup_batches 1000 \
  --grad_clip 1.0 \
  --loss arcface \
  --num_classes 85742 \
  --arcface_margin 0.5 \
  --arcface_scale 64.0 \
  --amp \
  --compile \
  --num_workers 8 \
  --prefetch_factor 8 \
  --save_every 5 \
  --eval_batch_size 128 \
  --lfw_bin ./data/casia-webface/eval/lfw.bin \
  --cfp_fp_bin ./data/casia-webface/eval/cfp_fp.bin \
  --agedb_30_bin ./data/casia-webface/eval/agedb_30.bin \
  --output_dir "${OUTPUT_DIR}" \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
