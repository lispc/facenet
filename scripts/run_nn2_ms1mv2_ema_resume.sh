#!/usr/bin/env bash
# Resume from best.pth with EMA + low-LR semi-hard fine-tuning.
# A safer follow-up after hard negative mining caused immediate collapse.
set -euo pipefail

OUTPUT_DIR="./checkpoints/nn2_ms1mv2_lmdb_p64k16_ema_resume"
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
  --lr 1e-4 \
  --min_lr 1e-7 \
  --weight_decay 5e-4 \
  --scheduler cosine \
  --warmup_batches 0 \
  --grad_clip 1.0 \
  --mining semi-hard \
  --ema_decay 0.9999 \
  --reset_optimizer \
  --reset_scheduler \
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
