#!/usr/bin/env bash
# ResNet100-IR + ArcFace on MS1MV2 LMDB.
# 控制变量对照实验：只把实验 6 的 Triplet Loss 替换为 ArcFace，
# 其余配置（模型、数据、输入尺寸、embedding_dim、batch size、优化器等）尽量保持不变。
#
# 初始化：
# - 从 ResNet100 Triplet best checkpoint 恢复 backbone。
# - ArcFace head 随机初始化。
# - Optimizer / Scheduler / EMA 全部重置。
# - 第 1 epoch 冻结 backbone，只训练 ArcFace head；
#   第 2 epoch 起解冻全部参数 fine-tune。
set -euo pipefail

OUTPUT_DIR="./checkpoints/resnet100_ms1mv2_lmdb_p32k8_arcface_freeze1"
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
  --lr 1e-4 \
  --min_lr 1e-7 \
  --weight_decay 5e-4 \
  --scheduler cosine \
  --warmup_batches 0 \
  --grad_clip 1.0 \
  --loss arcface \
  --num_classes 85742 \
  --arcface_margin 0.5 \
  --arcface_scale 64.0 \
  --freeze_backbone_epochs 1 \
  --ema_decay 0.9999 \
  --reset_optimizer \
  --reset_scheduler \
  --reset_ema \
  --amp \
  --compile \
  --use_checkpoint \
  --num_workers 8 \
  --prefetch_factor 8 \
  --save_every 5 \
  --eval_batch_size 128 \
  --resume ./checkpoints/resnet100_ms1mv2_lmdb_p32k8_30ep_triplet/best.pth \
  --lfw_bin ./data/casia-webface/eval/lfw.bin \
  --cfp_fp_bin ./data/casia-webface/eval/cfp_fp.bin \
  --agedb_30_bin ./data/casia-webface/eval/agedb_30.bin \
  --output_dir "${OUTPUT_DIR}" \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
