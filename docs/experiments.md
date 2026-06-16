# FaceNet 复现实验记录

> 本仓库主要实验日志。所有模型均为 NN2（Inception 风格），输入 224×224，embedding_dim=128，训练数据为 MS1MV2（5.8M 图 / 85,742 人），评测使用 InsightFace `.bin` 协议。

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

## 下一步

- **实验 6**：把 backbone 从 NN2 换成 **ResNet100-IR**，其余（Triplet Loss、数据、训练参数、224×224、128-D embedding）保持不变，做单变量对照，确认 backbone 容量的影响。
- 若 ResNet100 + Triplet 显著超越 NN2，再在其基础上切换 ArcFace，与公开指标对齐。
