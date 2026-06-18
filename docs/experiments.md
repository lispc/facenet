# FaceNet 复现实验记录

> 本仓库主要实验日志。默认输入 224×224，embedding_dim=128，训练数据为 MS1MV2（5.8M 图 / 85,742 人），评测使用 InsightFace `.bin` 协议；模型包括 NN2（Inception 风格）与 ResNet100-IR。

---

## 环境速查

- 4 × RTX 3090 24 GB
- Python 3.13.12，PyTorch 2.12.0+cu126，CUDA 12.9
- 训练：`torchrun --nproc_per_node=4 train.py ...`
- 数据格式：MS1MV2 LMDB（已 `--preload_lmdb`）

---

## 实验 1：NN2 30 epoch semi-hard baseline

| 配置 | 值 |
|------|-----|
| Model | NN2 |
| Input size | 224×224 |
| P / K | 64 / 16（batch=1024）|
| Epochs | 30 |
| Loss | Triplet Loss + semi-hard mining |
| Margin | 0.2 |
| Optimizer | AdamW，lr=1e-3，weight_decay=5e-4 |
| Scheduler | cosine，min_lr=1e-6 |
| AMP + torch.compile | 是 |

**最佳结果（best.pth）**：

| 评测集 | Accuracy |
|--------|----------|
| LFW(bin) | 97.58% |
| CFP-FP | 87.17% |
| AgeDB-30 | 83.27% |

该 checkpoint 作为后续所有续训的公共起点。

---

## 实验 2：Hard negative mining 续训（失败）

- 从实验 1 的 `best.pth` 恢复，mining 改为 `hard`，继续训练 epochs 25–50。
- 一个 epoch 后 embedding 崩溃：`d_an≈0`，`d_ap≈0.001`，LFW 跌至 ~48%。
- 结论：直接切换为 hard mining 在该配置下不稳定，已 early stop。

---

## 实验 3：EMA + 低 LR semi-hard 续训

| 配置 | 值 |
|------|-----|
| 起点 | 实验 1 `best.pth` |
| LR | 1e-4 |
| min_lr | 1e-7 |
| Mining | semi-hard |
| EMA decay | 0.9999 |
| Optimizer / Scheduler | 重置 |
| Epochs | 50 |

**最佳结果**：

| 评测集 | Accuracy | 备注 |
|--------|----------|------|
| LFW(bin) | **97.65%** | Epoch 3 |
| CFP-FP | 87.09% | Epoch 3 |
| AgeDB-30 | 82.87% | Epoch 3 |

Epoch 4 LFW 回落至 97.52%，整体进入平台期，因此停止并切换为 ArcFace 做严格对照实验。

---

## 实验 4：ArcFace naive 对照实验（失败）

**目标**：只把 Triplet Loss 替换为 ArcFace，其余数据/模型/优化器/LR/batch size 等保持不变，精确比较两种损失。

### 初始化细节

- **Backbone**：从实验 1 的 `best.pth` 加载 NN2 骨架权重（Triplet 训练得到）。
- **ArcFace head**：全新随机初始化。
  - 权重矩阵 `W` shape 为 `(85742, 128)`。
  - 使用 `nn.init.xavier_uniform_(self.weight)`。
- **Optimizer / Scheduler / EMA**：全部重置，不继承 Triplet 训练状态。
- Backbone 不冻结，所有参数一起训练。

对应代码：
- `src/losses/arcface.py`：ArcFace 损失实现。
- `train.py`：新增 `--loss arcface`、`--num_classes`、`--arcface_margin`、`--arcface_scale`、`--reset_ema`；从 Triplet checkpoint 切换到 ArcFace 时自动使用 `strict=False` 加载 backbone。

### 配置

| 配置 | 值 |
|------|-----|
| Model | NN2 |
| Input size | 224×224 |
| P / K | 64 / 16（batch=1024）|
| Loss | ArcFace |
| num_classes | 85,742 |
| ArcFace margin | 0.5 |
| ArcFace scale | 64.0 |
| Optimizer | AdamW，lr=1e-4，weight_decay=5e-4 |
| Scheduler | cosine，min_lr=1e-7 |
| EMA decay | 0.9999 |
| AMP + torch.compile | 是 |
| Epochs | 50 |

### 结果

| Epoch | LFW(bin) | CFP-FP | AgeDB-30 |
|-------|----------|--------|----------|
| 1 | 88.73% | 66.74% | 73.42% |
| 2 | 81.75% | 61.27% | 69.52% |
| 3 | 79.32% | 56.80% | 69.42% |
| 4 | 76.45% | 57.46% | 68.25% |
| 5 | 75.35% | 56.21% | 64.65% |
| 6 | 72.73% | 55.79% | 61.30% |
| 7 | 72.25% | 55.53% | 58.27% |
| 8 | 69.73% | 57.27% | 57.47% |

- 从 Epoch 2 开始，LFW/CFP-FP/AgeDB-30 持续下降。
- 原因：随机初始化的分类头在 lr=1e-4 下梯度较大，很快就破坏了 Triplet 预训练 backbone 的 embedding 分布。
- 结论：naive 地从 Triplet 切换到 ArcFace 不可行，需要先让 head 在固定 backbone 上学习。

---

## 实验 5：ArcFace freeze-backbone 5 epochs（成功）

**目标**：在实验 4 完全相同配置下，增加 `--freeze_backbone_epochs 5`：
- 前 5 epoch 只训练 ArcFace head，backbone 固定为 Triplet best 权重；
- 第 6 epoch 起解冻 backbone，一起 fine-tune。

### 训练曲线关键节点

| Epoch | Train loss | LR | LFW(bin) | CFP-FP | AgeDB-30 | 备注 |
|-------|------------|----|----------|--------|----------|------|
| 1 | 42.78 | 9.99e-5 | **97.58%** | 87.11% | 83.47% | backbone 冻结，head 开始学习 |
| 2 | 40.67 | 9.96e-5 | 97.58% | 87.11% | 83.47% | head 继续拟合 |
| 3 | 39.58 | 9.91e-5 | 97.58% | 87.11% | 83.47% | 验证持平 |
| 4 | 38.84 | 9.84e-5 | 97.58% | 87.11% | 83.47% | 验证持平 |
| 5 | 38.30 | 9.76e-5 | 97.58% | 87.11% | 83.47% | 冻结期结束 |
| 6 | 38.73 | 9.65e-5 | 97.55% | 86.73% | 83.18% | 刚解冻，backbone 微扰 |
| 7 | 38.39 | 9.52e-5 | 97.50% | 87.13% | 83.20% | 指标回升 |
| 8 | 38.19 | 9.38e-5 | **97.65%** | 86.90% | 83.30% | **刷新 best** |
| 9 | 38.01 | 9.22e-5 | 97.47% | 86.93% | 83.12% | 波动 |
| 10 | 37.84 | 9.05e-5 | 97.52% | 86.91% | 83.47% | 波动 |
| 11 | 37.69 | 8.85e-5 | 97.50% | 86.79% | 83.28% | 平台期 |
| 12 | 37.54 | 8.65e-5 | 97.47% | 86.79% | 83.32% | 平台期 |
| 13 | 37.40 | 8.42e-5 | — | — | — | 继续训练中 |

### 最终结果（best）

| 评测集 | Accuracy | 与 Triplet 基线对比 |
|--------|----------|---------------------|
| LFW(bin) | **97.65% ± 0.37%** | ↑ 0.07%（基线 97.58%）|
| CFP-FP | 86.90% ± 1.03% | ↓ 0.27%（基线 87.17%）|
| AgeDB-30 | 83.30% ± 1.07% | ↑ 0.03%（基线 83.27%）|

- `best.pth` 位于 `checkpoints/nn2_ms1mv2_lmdb_p64k16_arcface_freeze5/best.pth`（Epoch 8）。
- Freeze backbone 策略有效保护了 Triplet 预训练表征，head 在前 5 epoch 内没有破坏 backbone。
- 解冻后 LFW 首次超过 semi-hard Triplet 基线，但 CFP-FP 没有同步提升，整体进入平台期。

### 关键结论

- 在 NN2 这个规模的 backbone 上，ArcFace 相比 Triplet semi-hard 能带来小幅提升（LFW +0.07%），但收益有限。
- 公开论文中 ArcFace 能达到 LFW 99.8%+，核心差距来自 **backbone 容量**（ResNet100-IR vs NN2）和 **训练强度**（SGD lr=0.1 + 多轮 decay + 更长迭代）。

---

## 关键结论

1. 公开 MS1MV2 + Triplet semi-hard 在 NN2 224×224 上可达到 LFW **97.65%**，与论文 99.63% 仍有差距，主要因为 backbone 容量、训练数据规模和损失差异。
2. Hard negative mining 在该实现下直接切换会导致崩溃，需要更谨慎的调度或更大的 batch。
3. ArcFace 作为更现代的人脸识别损失，在 NN2 上仅带来边际提升；要验证 ArcFace 的真正优势，需要换用更强的 backbone。

---

## 实验 6：ResNet100-IR + Triplet 单变量对照（已完成）

**目标**：只把 backbone 从 NN2 换成 ResNet100-IR，其余（Triplet Loss、MS1MV2 数据、224×224、128-D embedding、AdamW lr=1e-3、cosine scheduler 等）保持不变，验证 backbone 容量的影响。

### 实现

- 新增 `src/models/iresnet.py`：InsightFace 风格 IResNet50 / IResNet100（pre-activation + PReLU），输出 L2 归一化 128-D embedding。
- 在 `train.py` 注册 `iresnet50` / `iresnet100`。
- 为 IResNet 增加 `--use_checkpoint` 梯度检查点，以在 4×RTX 3090 24GB 上训练 52M 参数的 ResNet100。

### 显存限制与配置调整

| 项目 | NN2 baseline | ResNet100 目标 | 实际可运行 |
|------|--------------|----------------|------------|
| P / K | 64 / 16 | 64 / 16 | **32 / 8** |
| 全局 batch | 1024 | 1024 | **256** |
| 每 GPU batch | 256 | 256 | **32** |
| 每 epoch steps | 1000 | 1000 | **4000** |
| 每 epoch 样本数 | ~1.02M | ~1.02M | **~1.02M** |
| Gradient checkpointing | 否 | 否 | **是** |

- P=64 K=16 / P=32 K=16 在 24GB 上均 OOM。
- 通过打开 `--use_checkpoint` 并将 P/K 降到 32/8，同时把 step 数×4，保持每 epoch 总样本数与 NN2 baseline 相同。

### 运行命令

```bash
bash scripts/run_resnet100_ms1mv2_triplet.sh
```

- 输出目录：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_30ep_triplet`
- 日志：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_30ep_triplet/train.log`

### 训练结果

训练在第 7 个 epoch 后停止（后续切换为 ArcFace 做对照）。关键节点如下：

| Epoch | Train loss | valid_triplet_frac | LR | LFW(bin) | CFP-FP | AgeDB-30 | 备注 |
|-------|------------|--------------------|----|----------|--------|----------|------|
| 1 | 0.1018 | 99.94% | 1.00e-3 | 96.82% | 86.14% | 84.47% | 初始 warmup 中 |
| 2 | 0.0843 | 99.89% | 9.96e-4 | 97.62% | 89.16% | 87.25% | 大幅提升 |
| 3 | 0.0772 | 99.98% | 9.83e-4 | **98.13%** | 88.93% | 89.03% | **LFW best** |
| 4 | 0.0772 | 99.98% | 9.63e-4 | 98.08% | 89.86% | 89.78% | 平台期 |
| 5 | 0.0768 | 99.97% | 9.36e-4 | 98.10% | **90.00%** | 90.05% | CFP/AgeDB 继续涨 |
| 6 | 0.0771 | 99.97% | 9.01e-4 | 97.98% | 89.86% | 90.50% | AgeDB 继续涨 |
| 7 | 0.0772 | 99.98% | 8.78e-4 | 98.33% | **90.33%** | **90.00%** | LFW 反弹，CFP best |

- `best.pth` 位于 `checkpoints/resnet100_ms1mv2_lmdb_p32k8_30ep_triplet/best.pth`（Epoch 7，按 LFW）。
- ResNet100-IR 相较 NN2 带来显著提升：LFW 从 97.65% → **98.33%**，CFP-FP 从 87.17% → **90.33%**，AgeDB-30 从 83.27% → **90.00%**。
- K=8 的负样本池确实使收敛更慢，但 7 个 epoch 后仍在稳步上涨；继续训练可能还有收益，但按计划在此时切换损失函数。

### 风险回顾

- K=8 导致每个 anchor 只有 7 个 negatives，semi-hard mining 的负样本池变小，收敛速度比大 batch 慢。
- Gradient checkpointing + torch.compile 使每步约 2.1 s，7 epoch 约 16.5 小时。

---

## 实验 7：ResNet100-IR + ArcFace freeze-backbone 1 epoch 续训（已放弃）

**目标**：在实验 6 的 ResNet100-IR Triplet 权重基础上，只把损失替换为 ArcFace，其余尽量保持不变；第 1 epoch 冻结 backbone 只训练 head，第 2 epoch 起解冻全部参数 fine-tune。

**放弃原因**：训练过程中 Epoch 2 解冻 backbone 后 LFW 从 98.33% 跌至 74.38%，train loss 虽然下降但验证指标未恢复；用户决定改为做 **Triplet vs ArcFace 的严格单变量对照实验**（实验 8），因此停止本实验。

### 初始化细节

- **Backbone**：从 `checkpoints/resnet100_ms1mv2_lmdb_p32k8_30ep_triplet/best.pth` 加载 ResNet100-IR 权重。
- **ArcFace head**：全新随机初始化（`W` shape `(85742, 128)`，Xavier uniform）。
- **Optimizer / Scheduler / EMA**：全部重置，不继承 Triplet 训练状态。
- **冻结策略**：`--freeze_backbone_epochs 1`，第 1 epoch 只训练 ArcFace head，第 2 epoch 起所有参数一起训练。

### 配置

| 配置 | 值 |
|------|-----|
| Model | ResNet100-IR (`iresnet100`) |
| Input size | 224×224 |
| P / K | 32 / 8（全局 batch 256）|
| num_batches_per_epoch | 4000 |
| Loss | ArcFace |
| num_classes | 85,742 |
| ArcFace margin | 0.5 |
| ArcFace scale | 64.0 |
| Optimizer | AdamW，lr=1e-4，weight_decay=5e-4 |
| Scheduler | cosine，min_lr=1e-7 |
| EMA decay | 0.9999 |
| Freeze backbone | 1 epoch |
| AMP + torch.compile + gradient checkpointing | 是 |
| Epochs | 30 |

### 运行命令

```bash
bash scripts/run_resnet100_ms1mv2_arcface_freeze1.sh
```

- 输出目录：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_arcface_freeze1`
- 日志：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_arcface_freeze1/train.log`

### 预期

- 相比 NN2 + ArcFace freeze5（LFW 97.65%），ResNet100-IR 更强的 backbone 应能进一步逼近公开 ArcFace/InsightFace 报告的 99%+。
- 第 1 epoch 冻结 backbone 可防止随机初始化的分类头破坏 Triplet 预训练表征。

---

## 实验 8：ResNet100-IR + ArcFace（from scratch）单变量对照（已停止）

**目标**：与实验 6 的 ResNet100-IR + Triplet 做严格单变量对照，**仅把 loss 从 Triplet 替换为 ArcFace**，其余配置（网络、数据、输入尺寸、embedding_dim、batch size、优化器、LR、scheduler、epochs 等）完全一致。

### 配置

| 配置 | 值 |
|------|-----|
| Model | ResNet100-IR (`iresnet100`) |
| Input size | 224×224 |
| Embedding dim | 128 |
| P / K | 32 / 8（全局 batch 256）|
| num_batches_per_epoch | 4000 |
| Epochs | 30 |
| Optimizer | AdamW，lr=1e-3，weight_decay=5e-4 |
| Scheduler | cosine，min_lr=1e-6 |
| Warmup | 1000 batches |
| Gradient clip | 1.0 |
| AMP + torch.compile + gradient checkpointing | 是 |

### 仅不同的配置

| 配置 | Triplet | ArcFace |
|------|---------|---------|
| Loss | `--loss triplet --mining semi-hard` | `--loss arcface --num_classes 85742 --arcface_margin 0.5 --arcface_scale 64.0` |
| 初始化 | from scratch | from scratch |
| Resume | 无 | 无 |
| EMA | 未启用 | 未启用 |
| Freeze backbone | 无 | 无 |

### 运行命令

```bash
bash scripts/run_resnet100_ms1mv2_arcface_scratch.sh
```

- 输出目录：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_30ep_arcface`
- 日志：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_30ep_arcface/train.log`

### 结果

| 阶段 | Train loss | Train acc | LFW(bin) | CFP-FP | AgeDB-30 | 备注 |
|------|------------|-----------|----------|--------|----------|------|
| Epoch 1 | 43.86 | 0.00% | **52.88% ± 2.03%** | 47.96% ± 1.11% | 47.70% ± 1.23% | head 刚开始学习 |

- 与 Triplet 实验 Epoch 1 的 LFW **96.82%** 相比，ArcFace from scratch 起点明显更低，符合随机初始化分类头 + 小 batch 的预期。
- 已停止，进入实验 9/10/11 尝试更大 batch 与 SGD。

---

## 实验 9：ResNet100-IR + ArcFace from scratch，effective batch 4× + lr 线性放大（已停止）

**目标**：在实验 8 完全相同配置的基础上，**只改两个参数**：
- `--accum_steps 4`：把 effective global batch 从 1024 放大到 4096（每卡 micro-batch 仍为 256，显存不变）；
- `--lr 4e-3`：按 linear scaling rule 把 AdamW 学习率同步放大 4 倍。

其他所有参数（模型、数据、P/K、epochs、scheduler、loss 超参等）与实验 8 保持一致。

### 运行命令

```bash
bash scripts/run_resnet100_ms1mv2_arcface_accum4.sh
```

- 输出目录：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_accum4_arcface`
- 日志：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_accum4_arcface/train.log`

### 结果

| 阶段 | Train loss | Train acc | LFW(bin) | CFP-FP | AgeDB-30 | 备注 |
|------|------------|-----------|----------|--------|----------|------|
| Epoch 1 | 46.38 | 0.00% | 51.67% ± 1.08% | **63.66% ± 1.75%** | 50.10% ± 1.83% | CFP-FP 提升明显，LFW 未超越实验 8 |

- CFP-FP 从实验 8 的 47.96% 提升到 63.66%，说明大 batch + 高 LR 对 front-profile 有帮助。
- 但 LFW 仍只有 51.67%，train loss 没有更快下降，acc 仍为 0%，整体未达预期。
- 已停止本实验，准备切换为 SGD + step decay + lr=0.1（实验 10）。

## 实验 10：ResNet100-IR + ArcFace from scratch，SGD + lr=0.1 + step decay（已停止）

**目标**：在实验 9 的基础上，**只改优化器和学习率调度**：
- `--optimizer sgd`（momentum=0.9）
- `--lr 1e-1`
- `--scheduler step`（在总步数 50% 和 80% 处 lr ×0.1）

其余所有参数（模型、数据、P/K、`accum_steps=4`、loss 超参、epochs 等）与实验 9 保持一致。

### 运行命令

```bash
bash scripts/run_resnet100_ms1mv2_arcface_sgd.sh
```

- 输出目录：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_sgd_arcface`
- 日志：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_sgd_arcface/train.log`

### 结果

| 阶段 | Train loss | Train acc | LFW(bin) | CFP-FP | AgeDB-30 | 备注 |
|------|------------|-----------|----------|--------|----------|------|
| Epoch 1 | 55.77 | 0.00% | **63.60% ± 1.88%** | 54.43% ± 1.47% | 47.55% ± 1.43% | LFW 比 AdamW 好 |
| Epoch 2 | 49.79 | 0.00% | **54.38% ± 1.95%** | 54.29% ± 1.08% | 48.12% ± 1.75% | LFW 回落，整体停滞 |

- SGD + lr=0.1 能让 train loss 下降（55.77 → 49.79），但验证指标没有持续提升。
- 主要问题：仍用 224×224 输入、128-D embedding、effective batch 4096，和标准 ArcFace 配置差距较大。
- 已停止本实验，准备按标准 ArcFace 配置重启（实验 11）。

## 实验 11：ResNet100-IR + ArcFace 标准对齐（已结束）

**目标**：让配置尽可能接近 ArcFace / InsightFace 论文标准，核心改动：
- 输入尺寸：`224×224` → **112×112**
- Embedding dim：`128` → **512**
- Global batch：`4096`（accum=4） → **512**（无 accum，每卡 64×2=128）
- 关闭 gradient checkpointing
- 保持 SGD lr=0.1、step scheduler、约 180k 总步数

### 与标准 ArcFace 的对齐情况

| 配置 | 标准 ArcFace | 实验 11 |
|------|--------------|---------|
| Backbone | ResNet100-IR | ResNet100-IR |
| 输入尺寸 | 112×112 | **112×112** |
| Embedding dim | 512 | **512** |
| Global batch | 512 | **512** |
| Optimizer | SGD, momentum=0.9, wd=5e-4 | SGD, momentum=0.9, wd=5e-4 |
| Initial LR | 0.1 | **0.1** |
| LR decay | step, 100k/160k iters | step（代码默认 50%/80% 总步数） |
| 总步数 | ~180k | **~182k**（11376 steps × 16 epochs） |
| Dropout | ~0.5 | 0.5 |

### 运行命令

```bash
bash scripts/run_resnet100_ms1mv2_arcface_standard.sh
```

- 输出目录：`checkpoints/resnet100_ms1mv2_lmdb_p64k2_16ep_arcface_standard`
- 日志：`checkpoints/resnet100_ms1mv2_lmdb_p64k2_16ep_arcface_standard/train.log`

### 结果

| Epoch | LFW | CFP-FP | AgeDB-30 | 备注 |
|------:|----:|-------:|---------:|------|
| 1 | 90.68% ± 0.75% | 77.46% ± 2.00% | 68.10% ± 1.43% | 起点 |
| 2 | 96.07% ± 0.73% | 82.57% ± 1.01% | 78.33% ± 1.90% | — |
| 3 | 97.23% ± 0.53% | 82.34% ± 1.14% | 86.60% ± 1.10% | — |
| 4 | 98.20% ± 0.36% | 85.29% ± 0.87% | 88.07% ± 1.37% | — |
| 5 | 98.52% ± 0.28% | 86.19% ± 1.13% | 89.03% ± 0.91% | — |
| 6 | 98.58% ± 0.40% | 88.20% ± 0.85% | 89.73% ± 1.46% | — |
| 7 | 98.78% ± 0.33% | 88.24% ± 1.01% | 89.47% ± 0.90% | — |
| 8 | 98.72% ± 0.24% | 87.81% ± 1.47% | 89.70% ± 1.05% | LFW 略降，CFP 略降，AgeDB 微升 |
| 9 | 99.43% ± 0.30% | 92.77% ± 0.94% | 93.78% ± 0.74% | lr 降到 0.01 后全面暴涨 |
| 10 | 99.32% ± 0.37% | 93.26% ± 0.93% | 94.50% ± 0.92% | LFW 微降，CFP/AgeDB 继续提升 |
| 11 | 99.52% ± 0.30% | 93.56% ± 0.66% | 94.80% ± 0.71% | 三项全部刷新 best |
| 12 | 99.52% ± 0.22% | 93.70% ± 0.67% | 94.92% ± 0.60% | LFW 持平，CFP/AgeDB 继续微升 |
| 13 | 99.50% ± 0.26% | 94.56% ± 0.58% | 95.65% ± 0.47% | lr 降到 1e-3，CFP/AgeDB 大幅提升 |

- Epoch 8 平均 loss：33.98，lr 1.00e-1；Epoch 9 平均 loss：23.83，lr 1.00e-2。
- Epoch 10–13 平均 loss/acc：24.78/1.30%、16.41/1.93%、17.16/1.44%、17.02/1.54%。
- 与 Triplet best 对比：
  - Triplet：LFW 98.33% / CFP-FP 90.33% / AgeDB-30 90.00%
  - ArcFace Epoch 13：LFW 99.50% / CFP-FP 94.56% / AgeDB-30 95.65%（全面大幅超越）

### 最终状态

- **实验 11 已结束**，训练在 Epoch 14 约 6% 处被手动停止。
- **最佳 checkpoint**：`checkpoints/resnet100_ms1mv2_lmdb_p64k2_16ep_arcface_standard/best.pth`（Epoch 12，按 LFW 准确率保存）。
  - Epoch 12 指标：LFW **99.52%** / CFP-FP **93.70%** / AgeDB-30 **94.92%**
- **最后完成 epoch（Epoch 13）指标**：LFW **99.50%** / CFP-FP **94.56%** / AgeDB-30 **95.65%**
- 结论：在标准对齐的 ArcFace 配置下，ResNet100-IR 在 MS1MV2 上取得了远超 Triplet 的性能；输入 112×112、embedding 512、global batch 512、SGD+step decay 是关键配置。

### 项目结论

- **Input size 112×112 足够**：相比之前 224×224，速度更快、显存更省，且最终指标达到 LFW 99.5%+。
- **ArcFace 需要大 embedding dim 与大 batch**：之前 128-D / batch 256 的 ArcFace 实验失败；改为 512-D / batch 512 后性能突破。
- **SGD + lr=0.1 + step decay 是 ArcFace 的关键**：AdamW + cosine 在该数据上无法让 ArcFace head 从头学习。
- **lr decay 后的提升非常明显**：第一次 decay（0.1→0.01）后 Epoch 9 暴涨；第二次 decay（0.01→0.001）后 Epoch 13 CFP/AgeDB 继续提升。
- 项目到此结束，后续无需再跑 Epoch 14–16。
