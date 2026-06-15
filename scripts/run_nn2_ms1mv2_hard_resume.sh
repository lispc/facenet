#!/usr/bin/env bash
# Resume from best.pth and switch to hard negative mining for another 20 epochs.
# Goal: push LFW beyond the 97.58% plateau reached by semi-hard mining.
set -euo pipefail

OUTPUT_DIR="./checkpoints/nn2_ms1mv2_lmdb_p64k16_hard_resume"
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
  --epochs 50 \
  --optimizer adamw \
  --lr 1e-3 \
  --min_lr 1e-7 \
  --weight_decay 5e-4 \
  --scheduler cosine \
  --warmup_batches 0 \
  --grad_clip 1.0 \
  --mining hard \
  --amp \
  --compile \
  --num_workers 8 \
  --prefetch_factor 8 \
  --save_every 5 \
  --eval_batch_size 128 \
  --resume ./checkpoints/nn2_ms1mv2_lmdb_p64k16_30ep/best.pth \
  --lfw_bin ./data/casia-webface/eval/lfw.bin \
  --cfp_fp_bin ./data/casia-webface/eval/cfp_fp.bin \
  --agedb_30_bin ./data/casia-webface/eval/agedb_30.bin \
  --output_dir "${OUTPUT_DIR}" \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
