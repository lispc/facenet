# FaceNet / ArcFace 复现（PyTorch）

基于 Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering" (CVPR 2015) 的方法级复现，并扩展了 **ArcFace** 损失与 **ResNet100-IR** backbone 的对比实验。

## 环境

- Python 3.10+
- PyTorch 2.0+ with CUDA
- 本机实测：4 × RTX 3090 24 GB，128 核 CPU，995 GB RAM

```bash
pip install -r requirements.txt
```

## 项目结构

```
src/
  models/       # FaceNet 模型（NN2/NN3/NN4/NNS1/NNS2）与 ResNet100-IR（iresnet.py）
  losses/       # Triplet Loss、ArcFace Loss
  mining/       # Online triplet mining 策略
  data/         # Dataset（ImageFolder / .rec / LMDB）与 LFW 评测加载
  utils/        # 工具函数
scripts/        # 训练/数据转换/辅助脚本
train.py        # 训练脚本（DDP + AMP，支持 Triplet / ArcFace）
eval_lfw.py     # LFW pairs.txt 评测脚本
eval_bin.py     # InsightFace .bin 评测脚本（lfw.bin / cfp_fp.bin / agedb_30.bin）
```

## 快速开始

### 1. 准备数据

本机已通过 Kaggle 下载：

| 数据集 | 用途 | 原始大小 | LMDB 大小 | 样本数 |
|--------|------|----------|-----------|--------|
| MS1MV2 112×112 | 主训练集 | ~23 GB 解压图 | ~25 GB | ~5.8M |
| CASIA-WebFace 112×112 | baseline | ~2.7 GB `.rec` | ~4 GB | ~490K |
| LFW / CFP-FP / AgeDB-30 | 评测 | 含在数据集中 `.bin` | - | - |

#### 推荐：LMDB 格式

为避免训练时加载大量小图片的 I/O 瓶颈，数据集统一转换为 **LMDB**：

```bash
# MS1MV2 已经是 ImageFolder，直接转 LMDB
python scripts/folder_to_lmdb.py \
  --root ./data/ms1mv2/ms1m-arcface \
  --output ./data/ms1mv2.lmdb

# CASIA-WebFace 是 .rec，直接转 LMDB
python scripts/rec_to_lmdb_casia.py \
  --rec_root ./data/casia-webface/casia-webface \
  --output ./data/casia-webface.lmdb
```

#### 其他格式支持

- `--dataset_type imagefolder`：直接读解压后的图片文件夹。
- `--dataset_type mxrec`：直接读 `.rec` / `.idx`（不依赖 mxnet，纯 Python reader）。
- `--dataset_type lmdb`：读 LMDB（推荐，支持 `--preload_lmdb` 全量进内存）。

### 2. 训练

#### NN2 224×224 on MS1MV2（Triplet semi-hard）

```bash
bash scripts/run_nn2_ms1mv2_strong.sh
```

等价命令：

```bash
torchrun --nproc_per_node=4 train.py \
  --data_root ./data/ms1mv2.lmdb --dataset_type lmdb --preload_lmdb \
  --model nn2 --input_size 224 --p 64 --k 16 \
  --num_batches_per_epoch 1000 --epochs 30 \
  --optimizer adamw --lr 1e-3 --min_lr 1e-6 --weight_decay 5e-4 \
  --scheduler cosine --warmup_batches 1000 --grad_clip 1.0 \
  --mining semi-hard --amp --num_workers 8 \
  --save_every 5 --eval_batch_size 128 \
  --lfw_bin ./data/casia-webface/eval/lfw.bin \
  --cfp_fp_bin ./data/casia-webface/eval/cfp_fp.bin \
  --agedb_30_bin ./data/casia-webface/eval/agedb_30.bin \
  --output_dir ./checkpoints/nn2_ms1mv2_lmdb_p64k16_30ep
```

#### ResNet100-IR + ArcFace 标准配置 on MS1MV2（项目最终实验）

```bash
bash scripts/run_resnet100_ms1mv2_arcface_standard.sh
```

等价命令：

```bash
torchrun --nproc_per_node=4 train.py \
  --data_root ./data/ms1mv2.lmdb --dataset_type lmdb --preload_lmdb \
  --model iresnet100 --input_size 112 --embedding_dim 512 --dropout 0.5 \
  --p 64 --k 2 --num_batches_per_epoch 11376 --epochs 16 \
  --optimizer sgd --lr 1e-1 --min_lr 1e-6 --weight_decay 5e-4 \
  --scheduler step --warmup_batches 1000 --grad_clip 1.0 \
  --loss arcface --num_classes 85742 --arcface_margin 0.5 --arcface_scale 64.0 \
  --amp --compile --num_workers 8 --prefetch_factor 8 --save_every 5 \
  --eval_batch_size 128 \
  --lfw_bin ./data/casia-webface/eval/lfw.bin \
  --cfp_fp_bin ./data/casia-webface/eval/cfp_fp.bin \
  --agedb_30_bin ./data/casia-webface/eval/agedb_30.bin \
  --output_dir ./checkpoints/resnet100_ms1mv2_lmdb_p64k2_16ep_arcface_standard
```

### 3. 评测

使用 InsightFace `.bin` 文件：

```bash
python eval_bin.py \
  --checkpoint ./checkpoints/resnet100_ms1mv2_lmdb_p64k2_16ep_arcface_standard/best.pth \
  --bin_path ./data/casia-webface/eval/lfw.bin \
  --model iresnet100 --input_size 112 --embedding_dim 512
```

或使用 LFW `pairs.txt`：

```bash
python eval_lfw.py \
  --checkpoint ./checkpoints/nn2_ms1mv2/best.pth \
  --lfw_root ./data/lfw \
  --pairs ./data/lfw/pairs.txt \
  --model nn2 --input_size 224
```

## 实现要点

- **模型**：
  - FaceNet：NN2/NN3/NN4/NNS1/NNS2 Inception 风格 backbone，最后一层 128-D + L2 归一化。
  - ArcFace：InsightFace 风格 ResNet50-IR / ResNet100-IR，支持 128-D / 512-D embedding 与 Dropout。
- **损失**：Triplet Loss（margin α=0.2）与 ArcFace（margin=0.5，scale=64）。
- **Mining**：在线 semi-hard / hard negative mining，PK sampler 保证 batch 内 anchor-positive 对充足。
- **训练**：`torchrun` 4 卡 DDP + `torch.amp` 混合精度 + gradient accumulation + `torch.compile`。
- **数据格式**：LMDB（单一大文件，随机读快，适合多 worker），支持 `--preload_lmdb`。

## 实验结果

### 关键结论

1. **公开 MS1MV2 + NN2 + Triplet semi-hard**：LFW 可达 **97.65%**（论文 99.63%，差距主要来自 backbone 容量、训练数据规模和损失函数）。
2. **换用 ResNet100-IR + Triplet**：LFW **98.33%** / CFP-FP **90.33%** / AgeDB-30 **90.00%**，显著优于 NN2。
3. **标准 ArcFace 配置（ResNet100-IR、112×112、512-D、batch 512、SGD lr=0.1 + step decay）**：
   - 最佳 checkpoint（Epoch 12）：LFW **99.52%** / CFP-FP **93.70%** / AgeDB-30 **94.92%**
   - 最后完成 epoch（Epoch 13）：LFW **99.50%** / CFP-FP **94.56%** / AgeDB-30 **95.65%**
   - 全面大幅超越 Triplet，证明 **embedding dim 512 + batch 512 + SGD step decay** 是此前 ArcFace 实验失败的关键原因。

### 短周期 baseline（5 epochs）

| 模型 | 训练数据 | LFW | CFP-FP | AgeDB-30 | 备注 |
|------|----------|-----|--------|----------|------|
| NN4 96×96 | CASIA-WebFace | 87.40% ± 0.93% | 78.40% ± 1.76% | 68.95% ± 1.30% | 数据量小，明显欠拟合 |
| NN4 96×96 | MS1MV2 | 94.30% ± 0.79% | 82.26% ± 1.48% | 74.92% ± 1.31% | 数据量增加带来明显提升 |
| NN2 224×224 | MS1MV2 | 94.65% ± 0.70% | 83.31% ± 1.56% | 76.70% ± 1.34% | 模型/输入放大，小幅提升 |

> 完整实验记录、每次尝试的详细分析与教训见 `docs/experiments.md`。

## 当前状态

- ✅ 项目结构与依赖
- ✅ FaceNet 模型（NN2/NN3/NN4/NNS1/NNS2）+ L2 归一化
- ✅ ResNet50-IR / ResNet100-IR + ArcFace head
- ✅ Triplet Loss + ArcFace Loss
- ✅ semi-hard / hard online mining
- ✅ ImageFolder / MXNet `.rec` / LMDB 数据加载
- ✅ 纯 Python ImageRecordIO reader（不依赖 mxnet）
- ✅ 4 卡 DDP + AMP + `torch.compile` 训练脚本
- ✅ LFW pairs 评测 + InsightFace `.bin` 评测
- ✅ MS1MV2 / CASIA-WebFace 数据集下载并转 LMDB
- ✅ NN2 30 epoch Triplet semi-hard on MS1MV2：LFW **97.65%**
- ✅ ResNet100-IR 7 epoch Triplet on MS1MV2：LFW **98.33%** / CFP **90.33%** / AgeDB **90.00%**
- ✅ ResNet100-IR + ArcFace 标准对齐 on MS1MV2：LFW **99.52%** / CFP **93.70%** / AgeDB **94.92%**
- ✅ 项目已结束，文档已整理

## 实验计划与记录

- 总体计划与资源评估：`docs/facenet_reproduction_plan.md`
- 完整实验日志与结论：`docs/experiments.md`
