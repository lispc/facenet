# FaceNet 复现（PyTorch）

基于 Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering" (CVPR 2015) 的方法级复现。

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
  models/       # FaceNet 模型（NN2/NN3/NN4/NNS1/NNS2）
  losses/       # Triplet Loss
  mining/       # Online triplet mining 策略
  data/         # Dataset（ImageFolder / .rec / LMDB）与 LFW 评测加载
  utils/        # 工具函数
scripts/        # 数据下载、格式转换、合成数据生成等辅助脚本
train.py        # 训练脚本（DDP + AMP）
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
# 注意：Kaggle 版 CASIA-WebFace 的 .rec 格式特殊（header label 字段实际为 payload 长度，
# 且图像前有 24 字节二次头），请使用专用转换脚本：
python scripts/rec_to_lmdb_casia.py \
  --rec_root ./data/casia-webface/casia-webface \
  --output ./data/casia-webface.lmdb
```

#### 其他格式支持

- `--dataset_type imagefolder`：直接读解压后的图片文件夹。
- `--dataset_type mxrec`：直接读 `.rec` / `.idx`（不依赖 mxnet，纯 Python reader）。
- `--dataset_type lmdb`：读 LMDB（推荐，支持 `--preload_lmdb` 全量进内存）。

### 2. 训练

#### NN4 96×96 baseline on CASIA-WebFace

```bash
torchrun --nproc_per_node=4 train.py \
  --data_root ./data/casia-webface.lmdb \
  --dataset_type lmdb \
  --model nn4 \
  --input_size 96 \
  --p 32 --k 8 \
  --epochs 5 \
  --num_batches_per_epoch 2000 \
  --output_dir ./checkpoints/nn4_casia_lmdb \
  --amp --mining semi-hard \
  --lr 1e-3 --scheduler cosine
```

#### NN2 224×224 on MS1MV2

```bash
torchrun --nproc_per_node=4 train.py \
  --data_root ./data/ms1mv2.lmdb \
  --dataset_type lmdb \
  --model nn2 \
  --input_size 224 \
  --p 16 --k 8 \
  --epochs 5 \
  --num_batches_per_epoch 2000 \
  --output_dir ./checkpoints/nn2_ms1mv2_lmdb \
  --amp --mining semi-hard \
  --lr 1e-3 --scheduler cosine
```

#### NN2 224×224 on MS1MV2（更强配置：更大 batch + 更长周期）

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

### 3. 评测

使用 InsightFace `.bin` 文件（数据集自带）：

```bash
python eval_bin.py \
  --checkpoint ./checkpoints/nn4_casia/best.pth \
  --bin_path ./data/casia-webface/eval/lfw.bin \
  --model nn4 --input_size 96
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

- **模型**：NN2/NN3/NN4/NNS1/NNS2 Inception 风格 backbone，最后一层 128-D + L2 归一化。
- **损失**：Triplet Loss，margin α=0.2。
- **Mining**：在线 semi-hard / hard negative mining，PK sampler 保证 batch 内 anchor-positive 对充足。
- **训练**：`torchrun` 4 卡 DDP + `torch.amp` 混合精度 + gradient accumulation。
- **数据格式**：LMDB（单一大文件，随机读快，适合多 worker）。

## 实验结果

### Baseline（5 epochs，semi-hard mining）

| 模型 | 训练数据 | LFW | CFP-FP | AgeDB-30 | 备注 |
|------|----------|-----|--------|----------|------|
| NN4 96×96 | CASIA-WebFace | 87.40% ± 0.93% | 78.40% ± 1.76% | 68.95% ± 1.30% | 数据量小，明显欠拟合 |
| NN4 96×96 | MS1MV2 | 94.30% ± 0.79% | 82.26% ± 1.48% | 74.92% ± 1.31% | 数据量增加带来明显提升 |
| NN2 224×224 | MS1MV2 | 94.65% ± 0.70% | 83.31% ± 1.56% | 76.70% ± 1.34% | 模型/输入放大，小幅提升 |

> 论文 FaceNet 在私有 100M–200M 图上训练，LFW 达 99.63%。公开数据集复现通常需要 MS1MV2/MS1MV3 + 更长训练周期 + 调优。

### 进行中

- NN2 224×224 on MS1MV2，20 epochs 长周期训练（目标 LFW 97%+）。
- 已准备好更强配置：NN2 224×224 / P=64 K=16 / 30 epochs / LMDB 预加载 / 每 epoch 评测 LFW+CFP-FP+AgeDB-30，待当前 20-epoch 跑完后启动。

## 实验计划

详见 `docs/facenet_experiment_plan.md`。

## 当前状态

- ✅ 项目结构与依赖
- ✅ FaceNet 模型（NN2/NN3/NN4/NNS1/NNS2）+ L2 归一化
- ✅ Triplet Loss + semi-hard / hard online mining
- ✅ ImageFolder / MXNet `.rec` / LMDB 数据加载
- ✅ 纯 Python ImageRecordIO reader（不依赖 mxnet）
- ✅ 4 卡 DDP + AMP 训练脚本
- ✅ LFW pairs 评测 + InsightFace `.bin` 评测
- ✅ 合成数据 + MS1MV2 LMDB 冒烟测试通过
- ✅ CASIA-WebFace / MS1MV2 数据集已下载并转 LMDB
- ✅ CASIA-WebFace LMDB 标签修复与重转换
- ✅ NN4 baseline on CASIA-WebFace / MS1MV2（训练 + LFW/CFP-FP/AgeDB-30 评测）
- ✅ NN2 224×224 on MS1MV2 短周期训练与评测
- ⏳ NN2 224×224 on MS1MV2 20 epochs 长周期训练
- ⏳ 准备更强配置（P=64 K=16 / 30 epochs / LMDB preload）待启动
