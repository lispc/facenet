#!/usr/bin/env bash
# ResNet100-IR + ArcFace (from scratch) on MS1MV2 LMDB.
# 与 run_resnet100_ms1mv2_arcface_accum4.sh 唯一区别：
# - Optimizer: AdamW -> SGD (momentum=0.9)
# - LR: 4e-3 -> 1e-1
# - Scheduler: cosine -> step (在总步数 50%/80% 处 *0.1)
# 其余参数（模型、数据、batch、accum_steps、loss 超参等）保持一致。
set -euo pipefail

OUTPUT_DIR="./checkpoints/resnet100_ms1mv2_lmdb_p32k8_sgd_arcface"
mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node=4 train.py \
  --data_root ./data/ms1mv2.lmdb \
  --dataset_type lmdb \
  --preload_lmdb \
  --model iresnet100 \
  --input_size 224 \
  --embedding_dim 128 \
  --p 32 \
  --k 8 \
  --num_batches_per_epoch 4000 \
  --epochs 30 \
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
  --accum_steps 4 \
  --amp \
  --compile \
  --use_checkpoint \
  --num_workers 8 \
  --prefetch_factor 8 \
  --save_every 5 \
  --eval_batch_size 128 \
  --lfw_bin ./data/casia-webface/eval/lfw.bin \
  --cfp_fp_bin ./data/casia-webface/eval/cfp_fp.bin \
  --agedb_30_bin ./data/casia-webface/eval/agedb_30.bin \
  --output_dir "${OUTPUT_DIR}" \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
