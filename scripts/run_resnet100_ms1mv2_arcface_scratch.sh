#!/usr/bin/env bash
# ResNet100-IR + ArcFace (from scratch) on MS1MV2 LMDB.
# 控制变量对照实验：与 ResNet100-IR + Triplet 实验完全一致，仅 loss 不同。
#
# 对应 Triplet 脚本：scripts/run_resnet100_ms1mv2_triplet.sh
set -euo pipefail

OUTPUT_DIR="./checkpoints/resnet100_ms1mv2_lmdb_p32k8_30ep_arcface"
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
  --optimizer adamw \
  --lr 1e-3 \
  --min_lr 1e-6 \
  --weight_decay 5e-4 \
  --scheduler cosine \
  --warmup_batches 1000 \
  --grad_clip 1.0 \
  --loss arcface \
  --num_classes 85742 \
  --arcface_margin 0.5 \
  --arcface_scale 64.0 \
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
