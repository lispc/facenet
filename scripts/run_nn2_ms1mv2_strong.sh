#!/usr/bin/env bash
# Stronger NN2 224x224 training on MS1MV2 LMDB.
# Goals:
#   - larger effective batch (P=64, K=16 => 1024 per GPU, 4096 global)
#   - preload LMDB to RAM for stable I/O
#   - cosine LR with warmup and min_lr
#   - per-epoch LFW/CFP-FP/AgeDB-30 evaluation via InsightFace .bin files
#   - periodic checkpoints and resume support
set -euo pipefail

OUTPUT_DIR="./checkpoints/nn2_ms1mv2_lmdb_p64k16_30ep"
mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node=4 train.py \
  --data_root ./data/ms1mv2.lmdb \
  --dataset_type lmdb \
  --preload_lmdb \
  --model nn2 \
  --input_size 224 \
  --embedding_dim 128 \
  --p 64 \
  --k 16 \
  --num_batches_per_epoch 1000 \
  --epochs 30 \
  --optimizer adamw \
  --lr 1e-3 \
  --min_lr 1e-6 \
  --weight_decay 5e-4 \
  --scheduler cosine \
  --warmup_batches 1000 \
  --grad_clip 1.0 \
  --mining semi-hard \
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
