# FaceNet 复现实验计划（基于当前机器配置）

> 制定日期：2026-06-15  
> 依据：`docs/face_recognition_datasets_survey.md`、`docs/facenet_dataset_and_machine_recommendation.md`、`docs/facenet_reproduction_plan.md` 以及本机实际硬件信息。

---

## 1. 机器配置盘点

| 组件 | 实际配置 | 对实验的意义 |
|------|----------|--------------|
| GPU | **4 × NVIDIA RTX 3090 24 GB** | 可做多卡 DDP，单卡 24 GB 足够训练 NN2/NN3/NN4；配合 AMP 可尝试接近论文 batch 1,800 的 effective batch。 |
| CPU | 128 核 | 数据加载可设高 `num_workers`，预处理、LFW/YouTube 评测并行度充足。 |
| 内存 | 995 GB（可用 ~834 GB）| 可放心缓存 embeddings、做 offline/online mining，不用担心 OOM。 |
| 磁盘 | `/` 剩余 **~425 GB** | 可放下 MS1MV2 + VGGFace2 + LFW + YouTube Faces + 多个 checkpoint；Glint360K/WebFace42M 仍偏紧。 |
| 软件 | Python 3.13.12，PyTorch 2.12.0+cu126，4 GPU 可被 torch 识别 | 可直接使用 `torchrun` / `DistributedDataParallel` + `torch.amp`。 |

**结论**：本机属于「标准复现方案」的顶配甚至接近「论文级」门槛，不再受 33 GB 小磁盘约束，应该直接瞄准 **MS1MV2 + NN2/NN3 + LFW 99%+**，而不是只跑 CASIA 小模型。

---

## 2. 数据集选择

| 优先级 | 数据集 | 规模 | 预计占用 | 用途 |
|--------|--------|------|----------|------|
| ⭐ 首选训练集 | **MS1MV2 / MS1MV3 112×112** | ~5.2M–5.8M 图 / 85K–93K 人 | `.rec` ~16.5 GB；解压后 ~25 GB | 主力训练集，复现 FaceNet 方法并逼近论文 LFW 指标 |
| ⭐ 必备评测集 | **LFW** | 13,233 图 / 5,749 人 | ~180 MB | 论文主评测集，6000 对标准协议 |
| 冒烟/消融训练集 | **CASIA-WebFace 112×112** | ~494K 图 / 10.6K 人 | ~2.7 GB `.rec` | 快速验证 pipeline、NN4 baseline、超参扫描 |
| 可选评测集 | **YouTube Faces**（精简版） | 3,425 视频 / 1,595 人 | ~10 GB | 论文次评测集，有余力再补 |
| 可选对比训练集 | **VGGFace2** | 3.31M 图 / 9.1K 人 | 压缩 ~36 GB，解压 ~50–60 GB | 可做数据量/标签质量消融；磁盘仍够，但优先级低于 MS1MV2 |
| 不推荐当前做 | **WebFace42M / Glint360K** | 17M–42M 图 | 100+ GB | 能放下但训练周期长、与 FaceNet 论文方法关系不大，留到后续阶段 |

**存储预算（保守）**：MS1MV2 `.rec` 16.5 GB + 解压图 25 GB + LFW 0.2 GB + YouTube 10 GB + VGGFace2 60 GB + checkpoints/logs ~30 GB ≈ **140 GB**，远低于 425 GB 剩余。

---

## 3. 模型与训练策略

### 3.1 模型路线

| 模型 | 输入尺寸 | 参数量 | 定位 |
|------|----------|--------|------|
| **NN2（Inception）** | 224×224 | 7.5M | **主模型**，最接近论文大模型指标 |
| **NN3** | 160×160 | 类似 NN2 | 速度与精度的折中 |
| **NN4** | 96×96 | 类似 NN2 | 快速 baseline，在 CASIA 上跑通 pipeline |
| NNS1 / NNS2 | 165×165 / 140×116 | 26M / 4.3M | 可选做模型大小消融 |
| NN1（Zeiler&Fergus） | 220×220 | 140M | 参数过大，暂不作为首选，有余力再试 |

### 3.2 训练配置

- **分布式**：`torchrun --nproc_per_node=4 train.py`，4× RTX 3090 DDP。
- **混合精度**：`torch.cuda.amp`（或 `torch.amp`）自动切换，节省显存、加速 mining 阶段 embedding 计算。
- **Batch size 策略**：
  - 论文使用 batch 1,800（约 40 images/identity）。
  - 单卡 24 GB 对 224×224 NN2 大约能放 128–256 张图；4 卡 DDP 直接 global batch 512–1024。
  - 若仍想接近 1,800，可用 **gradient accumulation** 2–4 步，effective batch 达到 1k–2k。
- **优化器/学习率**：
  - 默认用 SGD / AdamW；论文用 SGD + AdaGrad，初始 lr 0.05。
  - 现代复现常用：AdamW + cosine decay / step decay；warmup 5k–10k steps。
- **Loss**：
  - 核心 **Triplet Loss**，margin α=0.2。
  - 先实现 **online semi-hard negative mining**，再升级到 hard negative mining。
  - 可额外跑 **ArcFace/Softmax** 作为对照（不属于 FaceNet 论文，但方便理解当前 SOTA baseline）。
- **数据增强**：随机水平翻转、颜色抖动、随机灰度、轻微模糊；输入从 112 resize 到目标尺寸。

---

## 4. 评测协议

1. **LFW 标准 6,000 对**
   - 计算 embedding 余弦距离/L2 距离；
   - 10-fold 交叉验证，报告 accuracy ± std；
   - 目标：**NN2 + MS1MV2 达到 99.0%–99.5%**。

2. **YouTube Faces（可选）**
   - 每段视频抽帧 → MTCNN 检测 → 取平均 embedding；
   - 报告 video-level accuracy。

3. **额外分析（消融）**
   - 不同输入分辨率（96 / 160 / 224）的精度-速度 trade-off；
   - 不同 embedding 维度（64 / 128 / 256 / 512）；
   - 不同 margin α（0.1 / 0.2 / 0.3）；
   - 训练数据量影响（MS1MV2 子采样 vs 全量）。

---

## 5. 分阶段执行计划（已更新为实际进度）

### Phase 0：环境与数据准备 ✅

- [x] 初始化代码仓库结构：`src/`, `scripts/`, `train.py`, `eval_lfw.py`, `eval_bin.py`。
- [x] 安装/确认依赖（见 `requirements.txt`）。
- [x] 下载 **CASIA-WebFace .rec**、**MS1MV2 .rec** 并转换为 **LMDB**。
- [x] 实现纯 Python 的 `.rec` 读取器（不依赖 mxnet）。
- [x] 4 卡 DDP + AMP 训练跑通。

### Phase 1：Baseline 跑通 ✅

- [x] NN4（96×96）+ Triplet Loss + online semi-hard mining。
- [x] CASIA-WebFace baseline：LFW 87.40%，CFP-FP 78.40%，AgeDB-30 68.95%。
- [x] MS1MV2 baseline（5 epochs NN4）：LFW 94.30%，CFP-FP 82.26%，AgeDB-30 74.92%。
- [x] NN2 224×224 short-run（5 epochs）：LFW 94.65%，CFP-FP 83.31%，AgeDB-30 76.70%。
- [x] LFW / CFP-FP / AgeDB-30 `.bin` 评测脚本完成。

### Phase 2：正式训练 MS1MV2 ⏳

- [x] 切到 **MS1MV2 + NN2 224×224**，4 卡 DDP + AMP。
- [x] 重写 semi-hard mining 为 O(N²·K) 内存，支持更大 batch。
- [ ] 当前 20-epoch 长周期训练进行中（预计 LFW 进一步提升）。
- [ ] 启动更强配置：`P=64 K=16`（effective batch 4096）、30 epochs、LMDB 预加载、每 epoch 评测。
- [ ] 目标：**LFW 99.0%–99.5%**。

### Phase 3：调优与消融（待 Phase 2 完成后）

- [ ] 调整 margin α、batch size、mining 策略（semi-hard → hard → batch-hard）。
- [ ] 尝试 NN3/NN4 在不同分辨率下的 trade-off。
- [ ] 可选：加入 VGGFace2 做跨数据集对比。
- [ ] 可选：下载并评测 YouTube Faces。

### Phase 4：工程化与报告（待 Phase 2 完成后）

- [x] 整理代码：模块化、README、一键训练/评测脚本。
- [ ] 保存并上传预训练权重、训练日志、TensorBoard curves。
- [ ] 撰写复现报告：与论文 Table 3/4/5/6 对比，分析差距原因。

**总预计周期**：约 **4–6 周**（已大幅提前，因数据下载和 baseline 验证已跑通）。

---

## 6. 关键风险与应对（已更新）

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 原训练集不公开，LFW 难达 99.63% | 无法完全复现论文数字 | 用 MS1MV2 做到 99.0%–99.5% 即视为方法级成功 |
| 4 卡 DDP 通信瓶颈 | RTX 3090 无 NVLink | AMP 降低带宽压力；已验证 4 卡 DDP 吞吐正常 |
| Online mining 收敛不稳定 | loss 震荡、精度上不去 | 已用 semi-hard mining + cosine lr + warmup； mining 内存优化后可上更大 batch |
| 224×224 batch 太大 OOM | 训练失败 | 已测 per-GPU N=1024 可用（~15.5 GB），N=2048 OOM；当前强配置使用 N=1024/卡 |
| 数据版权/合规 | MS1MV2 等非商业许可 | 仅用于学术研究，不外传原始数据和解包后的数据集 |

---

## 7. 近期下一步行动

1. 等当前 **NN2 224×224 20-epoch** 训练完成，评测最终 checkpoint。
2. 启动 **NN2 224×224 P=64 K=16 30-epoch** 强配置训练。
3. 根据 LFW/CFP/AgeDB 曲线继续调参（lr、weight decay、margin、mining）。
4. 整理预训练权重和复现报告。

---

## 8. 预期可交付成果

- 可复现的 PyTorch FaceNet 训练/评测代码库；
- NN4（CASIA）与 NN2（MS1MV2）的预训练权重；
- LFW 评测结果：NN2+MS1MV2 目标 **99.0%–99.5%**；
- 一份复现报告，包含与论文的对比、实现差异分析和消融实验。
