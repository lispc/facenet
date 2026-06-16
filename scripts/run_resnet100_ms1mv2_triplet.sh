#!/usr/bin/env bash
# ResNet100-IR + Triplet semi-hard on MS1MV2 LMDB.
# 控制变量对照实验：只把 backbone 从 NN2 换成 ResNet100-IR，
# 其余配置（数据、输入尺寸、embedding_dim、loss、优化器、学习率等）与 NN2 Triplet baseline 保持一致。
#
# 注意：ResNet100-IR 在 224×224 + P=64 K=16 下单卡 24GB 会 OOM；
# 即使 P=32 K=16（每 GPU batch 128）也会 OOM。
# 最终可稳定运行的最大配置为 P=32 K=8（每 GPU batch 32，全局 batch 256），
# 并打开 gradient checkpointing 以进一步节省显存。
# 为保持每 epoch 总样本数与 baseline 接近，将 step 数从 1000 提高到 4000
#（4000 × 256 = 1,024,000，与 NN2 baseline 的 1000 × 1024 相同）。
set -euo pipefail

OUTPUT_DIR="./checkpoints/resnet100_ms1mv2_lmdb_p32k8_30ep_triplet"
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
  --mining semi-hard \
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
