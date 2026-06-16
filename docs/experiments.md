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

## 实验 4：ArcFace 对照实验（进行中）

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

### 当前结果（naive 对照）

**Epoch 1**（从头训练 ArcFace head，backbone 已预训练）：

| 评测集 | Accuracy |
|--------|----------|
| LFW(bin) | 88.73% ± 1.02% |
| CFP-FP | 66.74% ± 1.17% |
| AgeDB-30 | 73.42% ± 2.10% |

说明：
- Train top-1 acc 在 pbar 上显示为 0.00%，因为类别数 85,742 很大，新 head 随机初始化时单个 batch 内几乎不可能命中正确类；但 head 正在快速学习。
- 第一个 epoch 后 LFW 已从随机头的 ~59% 恢复到 88.73%，说明 backbone 表征有效，head 正在拟合。

### 与基线对比

| 评测集 | Baseline（30ep semi-hard） | ArcFace Epoch 1 |
|--------|---------------------------|-----------------|
| LFW(bin) | 97.58% | 88.73% |
| CFP-FP | 87.17% | 66.74% |
| AgeDB-30 | 83.27% | 73.42% |

### 后续观察：naive 对照失败

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

### 下一步：freeze backbone 5 epochs

- 新脚本：`scripts/run_nn2_ms1mv2_arcface_freeze5.sh`
- 配置与 naive 对照完全相同，额外增加 `--freeze_backbone_epochs 5`：
  - 前 5 个 epoch 只训练 ArcFace head，backbone 保持 Triplet best 权重；
  - 第 6 个 epoch 开始解冻 backbone，一起 fine-tune。
- 预期：前 5 个 epoch 的验证指标应保持在 Triplet 基线（LFW 97.58%）附近，解冻后再观察是否进一步提升。

### Freeze5 Epoch 1（backbone frozen，只训 head）

| 评测集 | Accuracy | 与 Triplet 基线对比 |
|--------|----------|---------------------|
| LFW(bin) | **97.58% ± 0.40%** | 持平（基线 97.58%） |
| CFP-FP | 87.11% ± 1.24% | 持平（基线 87.17%） |
| AgeDB-30 | 83.47% ± 0.78% | 略高（基线 83.27%） |

- Train loss 从 naive 的 54.9 降到 42.8，说明 head 在固定 backbone 上学习更快。
- 验证指标没有下跌，证明 freeze backbone 策略有效保护了 Triplet 预训练表征。
- 后续继续关注 Epoch 6 解冻后的走势。

---

## 关键结论

1. 公开 MS1MV2 + Triplet semi-hard 在 NN2 224×224 上可达到 LFW **97.65%**，与论文 99.63% 仍有差距，主要因为训练数据规模（5.8M vs 100M–200M）和损失差异。
2. Hard negative mining 在该实现下直接切换会导致崩溃，需要更谨慎的调度或更大的 batch。
3. ArcFace 作为更现代的人脸识别损失，值得与 Triplet 做严格对照；当前第一个 epoch 已显示出快速收敛趋势。

---

## 下一步

- 继续跑完 ArcFace 50 epoch，监控 LFW / CFP-FP / AgeDB-30。
- 若 ArcFace 在 5–10 epoch 内显著超越 Triplet，则以其为基础进一步调优。
- 若仍平台化，考虑调整 ArcFace 的 margin/scale、学习率或增加 warmup。
