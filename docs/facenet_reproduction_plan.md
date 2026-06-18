# FaceNet 论文复现资源评估与计划

> 论文：FaceNet: A Unified Embedding for Face Recognition and Clustering  
> 作者：Florian Schroff, Dmitry Kalenichenko, James Philbin (Google Inc.)  
> 会议/时间：CVPR 2015 / arXiv:1503.03832

---

## 0. 当前实验状态

- 已实现 NN2/NN3/NN4/NNS1/NNS2 + Triplet Loss + semi-hard/hard mining。
- 在 MS1MV2（5.8M 图 / 85,742 人）上训练 NN2 224×224：
  - Triplet semi-hard 基线：LFW **97.58%**。
  - EMA + 低 LR semi-hard 续训：最高 LFW **97.65%**。
- 已完成 **ArcFace 对照实验**：
  - Naive 直接切换 ArcFace 会破坏 backbone，指标持续下跌。
  - 采用 **freeze backbone 5 epochs** 训练 ArcFace head 后，最终 best 为 LFW **97.65%**，小幅超过 Triplet 基线。
- **ResNet100-IR + Triplet 对照实验已完成 7 epoch**：
  - 新增 `src/models/iresnet.py`（IResNet50/100）并注册到 `train.py`。
  - 为 ResNet100 增加 `--use_checkpoint` 梯度检查点，使其在 4×RTX 3090 24GB 上可训练。
  - 由于显存限制，实际使用 **P=32 K=8，num_batches_per_epoch=4000**（全局 batch 256，每 epoch 约 1M 样本，与 NN2 baseline 总样本数相当）。
  - 运行脚本：`scripts/run_resnet100_ms1mv2_triplet.sh`。
  - 输出目录：`checkpoints/resnet100_ms1mv2_lmdb_p32k8_30ep_triplet`。
  - 7 epoch 最佳结果：**LFW 98.33% / CFP-FP 90.33% / AgeDB-30 90.00%**，显著优于 NN2。
- **ResNet100-IR + ArcFace 标准对齐实验已结束（项目最终实验）**：
  - 配置：112×112 输入、512-D embedding、global batch 512、SGD lr=0.1 + step decay、16 epochs。
  - 最佳 checkpoint：`checkpoints/resnet100_ms1mv2_lmdb_p64k2_16ep_arcface_standard/best.pth`（Epoch 12）。
  - 最佳指标：LFW **99.52%** / CFP-FP **93.70%** / AgeDB-30 **94.92%**。
  - 最后完成 epoch（Epoch 13）：LFW **99.50%** / CFP-FP **94.56%** / AgeDB-30 **95.65%**。
- **项目已结束**，详细实验日志与结论见 [`docs/experiments.md`](./experiments.md)。

---

## 1. 论文核心要点

- **目标**：学习一个从人脸图像到紧致欧氏空间（128-D）的映射，使得同一人的嵌入距离小、不同人的距离大。
- **网络**：
  - NN1：Zeiler&Fergus 风格 + 1×1 卷积，140M 参数，1.6B FLOPS。
  - NN2：Inception（GoogLeNet）风格，7.5M 参数，1.6B FLOPS。
  - NN3/NN4/NNS1/NNS2：更小/更低分辨率变体。
- **损失**：Triplet Loss（在线 hard/semi-hard negative mining），margin α=0.2。
- **训练**：
  - 数据：100M–200M 张人脸缩略图，约 800 万个身份。
  - 优化：SGD + AdaGrad，初始学习率 0.05。
  - batch：约 1,800 个样本/mini-batch，每个身份每批约 40 张脸。
  - 硬件：Google 内部 CPU 集群训练 1,000–2,000 小时（DistBelief）。
- **评测**：
  - LFW：99.63% ± 0.09（使用额外对齐）。
  - YouTube Faces：95.12% ± 0.39。
  - 内部 hold-out 与个人照片集合：VAL@FAR=10⁻³。

---

## 2. 复现资源评估

### 2.1 数据资源

| 数据集 | 论文使用 | 是否公开 | 规模 | 替代方案 |
|--------|----------|----------|------|----------|
| Google 内部人脸数据 | 训练 | **否** | 100M–200M 图 / ~8M 人 | MS-Celeb-1M、CASIA-WebFace、VGGFace2、WebFace42M 等 |
| LFW | 评测 | 是 | 13,233 图 / 5,749 人 | 直接下载 |
| YouTube Faces | 评测 | 是 | 3,425 视频 / 1,595 人 | 直接下载 |
| 内部 hold-out / 个人照片 | 评测 | 否 | ~1M 图 / ~12K 图 | 可用 LFW、IJB-B/C、CFP-FP 等公开评测集近似 |

**关键结论**：

1. **论文的 100M–200M 训练集不公开**，因此“像素级复现”原始指标几乎不可能。使用公开数据集（如 MS1M 或 VGGFace2）只能做到“方法级复现”，预期 LFW 会接近但不会完全达到 99.63%。
2. 公开数据集中，**MS-Celeb-1M（清洗后约 3.8M–5M 图）** 与 **CASIA-WebFace（约 500K 图）** 是最常用的 FaceNet 复现训练集。
3. 若只想跑通 pipeline、验证方法，**CASIA-WebFace 足够；若追求接近论文指标，建议 MS1M/VGGFace2 起步**。
4. 需要人脸检测/对齐模块（MTCNN 是公开实现中事实标准）。

### 2.2 计算资源

#### 原论文

- 平台：Google 内部 DistBelief CPU 集群。
- 时间：1,000–2,000 小时（约 42–83 天），模型规模较大。

#### 现代复现（GPU 估算）

| 模型 | 输入尺寸 | 参数量 | 单图 FLOPS | 推荐 GPU | 训练时间估算 |
|------|----------|--------|------------|----------|--------------|
| NN2 (Inception) | 224×224 | 7.5M | ~1.6B | A100 80GB / 多卡 A100 | 公开数据集上 3–7 天 |
| NN1 (Zeiler&Fergus) | 220×220 | 140M | ~1.6B | A100 80GB / 多卡 | 5–10 天或更长 |
| NN3/NN4 | 160×160 / 96×96 | 类似 NN2 | 更低 | RTX 4090 24GB / A100 | 1–3 天 |
| NNS1/NNS2 | 165×165 / 140×116 | 26M / 4.3M | 220M / 20M | 单卡 12GB+ | 几小时–1 天 |

注意点：

- **Batch size 1,800 对显存压力很大**。224×224 输入 + Inception 架构，单张前向约需数百 MB；1,800 张需要 50GB+ 显存，24GB 消费卡无法直接放下。
- 常见做法：
  - 使用 **gradient accumulation** 或 **multi-GPU data parallelism**；
  - 或 **减小 batch size**（如 256–512），论文指出 batch 越大 mining 效果越好，但现代实现可在小 batch 上通过 offline mining 弥补。
- 推断：单张 224×224 NN2 在现代 GPU 上 <10 ms，CPU 上也可达几十 ms。

#### 存储/内存

- 训练数据：
  - CASIA-WebFace 原图：~1–2 GB。
  - MS-Celeb-1M 原图：~30–50 GB。
  - VGGFace2 原图：~15 GB。
- 预处理（检测对齐后）缩略图：通常 5–20 GB。
- 模型 checkpoint：每个几十 MB 到几百 MB。
- 内存：建议训练机 ≥32 GB RAM；预处理/评测 16 GB 足够。

---

## 3. 复现计划

### 阶段 0：环境与数据准备（1–2 周）

1. **代码仓库初始化**
   - Python 3.10+、PyTorch（推荐）或 TensorFlow 2.x。
   - 依赖：`torch`, `torchvision`, `opencv-python`, `Pillow`, `numpy`, `scipy`, `scikit-learn`, `tqdm`, `tensorboard` 等。
   - 搭建日志与实验管理（TensorBoard / Weights & Biases）。

2. **数据下载**
   - 训练集（按优先级）：
     1. CASIA-WebFace（快速跑通）
     2. MS-Celeb-1M 清洗版（主流复现）
     3. VGGFace2（若需要更干净标签）
   - 评测集：
     - LFW（必备）
     - YouTube Faces（可选但推荐）
     - IJB-B / IJB-C / CFP-FP（可选，扩展评测）

3. **人脸检测与预处理**
   - 使用 MTCNN（`facenet-pytorch` 或 `insightface` 实现）检测并裁剪人脸。
   - 输出：论文中的“tight crop”，resize 到 96×96 / 160×160 / 224×224 等。
   - 保存预处理后的数据集索引（CSV/JSON），便于训练时快速读取。

### 阶段 1：Baseline 实现（2–3 周）

1. **模型架构**
   - 优先实现 **NN2（Inception）** 和/或 **NN4（96×96 轻量版）**，因为参数少、训练快。
   - 可选实现 NN1（Zeiler&Fergus）作为对比。
   - 关键点：
     - 最后一层输出 128-D；
     - L2 归一化；
     - ReLU / L2 pooling 细节与论文表格保持一致。

2. **Triplet Loss**
   - 公式实现：
     ```
     L = Σ max(||f(anchor) - f(positive)||² - ||f(anchor) - f(negative)||² + α, 0)
     ```
   - margin α=0.2。

3. **在线 Triplet Mining**
   - 先实现最基础版本：
     - 每个 batch 中同一身份取 anchor-positive 对；
     - 对每个 anchor-positive 对选 semi-hard negative（满足 d(a,p) < d(a,n) < d(a,p)+α）。
   - 验证 loss 能下降、VAL 能提升后再加入更复杂的 hard mining。

4. **训练脚本**
   - SGD/AdaGrad 或 AdamW（现代常用）+ 学习率衰减。
   - 支持 gradient accumulation / multi-GPU / mixed precision (AMP)。
   - 保存 checkpoint、记录 loss / VAL@FAR / LFW accuracy。

### 阶段 2：训练与调优（3–6 周）

1. **小规模验证**
   - 在 CASIA-WebFace 上先用 NN4（96×96）快速验证 pipeline。
   - 目标：LFW 达到 95%+ 作为 sanity check。

2. **正式训练**
   - 切到 MS1M / VGGFace2 + NN2/NN3。
   - 训练直到收敛（可能数百万步）。
   - 监控训练曲线，调整学习率、batch size、mining 策略。

3. **超参调优**
   - margin α：0.1–0.3；
   - batch size / 每个身份的样本数；
   - embedding 维度：128（默认），可试 64/256；
   - 数据增强：随机翻转、颜色抖动、模糊等。

### 阶段 3：评测与分析（2 周）

1. **LFW 评测**
   - 6000 对标准协议；
   - 计算余弦距离 / L2 距离，选最佳阈值；
   - 报告 accuracy 与 ROC。

2. **YouTube Faces 评测**
   - 每段视频取前 100/1000 帧做人脸检测；
   - 计算视频对之间的平均相似度，报告 accuracy。

3. **附加分析**
   - 不同模型大小/输入尺寸的 trade-off（复现 Figure 4/5）；
   - JPEG 质量、图像分辨率对性能的影响（复现 Table 4）；
   - embedding 维度影响（复现 Table 5）；
   - 训练数据量影响（复现 Table 6，子采样实验）。

### 阶段 4：工程化与复现报告（1–2 周）

1. **整理代码**
   - 模块化：data、model、loss、mining、train、eval。
   - 提供预训练权重下载、一键训练/评测脚本、README。

2. **撰写复现报告**
   - 与论文指标对比表；
   - 失败/未完全一致的原因分析（数据差异、实现细节、硬件差异）；
   - 给出可复现的 LFW/YouTube Faces 结果。

---

## 4. 主要风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 原训练集不公开 | 无法完全复现 99.63% | 使用 MS1M/VGGFace2，预期 LFW 99.0%–99.5% 区间 |
| Batch size 1,800 显存放不下 | 训练失败或 mining 质量差 | 多 GPU、gradient accumulation、或使用 offline mining 降低 batch 需求 |
| Online mining 实现复杂/收敛慢 | 训练不稳定 | 先用 offline mining 或小 batch semi-hard mining 验证；再逐步升级 |
| 人脸检测对齐差异 | 评测分数波动 | 固定 MTCNN，提供统一预处理脚本；与论文两种 LFW 评测模式对齐 |
| 训练时间成本 | 资源不足 | 先用 NN4/NNS1 在 CASIA 上验证；再决定是否上大规模训练 |

---

## 5. 推荐的最小可行复现路线（MVP）

如果资源有限，建议按以下路线快速验证核心方法：

1. 数据集：**CASIA-WebFace**（公开、小、快）。
2. 模型：**NN4**（96×96 Inception）或更小的 **NNS1/NNS2**。
3. GPU：**单卡 12GB+**（如 RTX 3060/4060）。
4. 训练时间：**1–3 天**。
5. 预期结果：LFW 95%–98%。

---

## 6. 资源清单速查

| 资源 | 最小配置 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3060 12GB | A100 40GB/80GB 或多卡 |
| CPU / RAM | 8 核 / 16 GB | 16 核 / 64 GB |
| 存储 | 50 GB SSD | 200 GB SSD（含 MS1M） |
| 训练数据 | CASIA-WebFace | MS-Celeb-1M 清洗版 |
| 网络 | 普通宽带 | 下载大数据集需要稳定网络 |
| 时间 | 1 周（MVP） | 1–2 个月（完整复现） |

---

## 7. 下一步行动

1. ✅ 已完成 NN2 + Triplet semi-hard 基线（LFW 97.58%）。
2. ✅ 已完成 NN2 + ArcFace 对照实验，freeze backbone 5 epochs best 为 LFW 97.65%。
3. ✅ 已完成 ResNet100-IR + Triplet 7 epoch 对照（LFW 98.33% / CFP-FP 90.33% / AgeDB-30 90.00%）。
4. **当前重点**：在 ResNet100-IR 上对比 **ArcFace vs Triplet**（实验 7，freeze backbone 1 epoch 后 fine-tune），向公开指标（LFW 99.7–99.8%）靠近。
5. 整理代码、脚本、预训练权重与复现报告。
