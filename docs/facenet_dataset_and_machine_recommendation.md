# FaceNet 复现：数据集与机器配置建议

> 基于 `face_recognition_datasets_survey.md` 的调研结论整理  
> 如果你准备换机器，以下配置和数据集选择可以直接参考。

---

## 1. 核心结论

- **论文原训练集不公开**：FaceNet 论文使用的是 Google 内部 100M–200M 张图、约 800 万个身份的数据。**99.63% 的 LFW 结果无法完全复现**。
- **公开数据集可达到 LFW 99.0%–99.7%**：其中 **MS1MV2 / MS1MV3** 是当前复现 FaceNet 最实用、最接近论文效果的训练集。
- **磁盘是最大约束**：原机只有 ~33 GB 剩余，放不下 VGGFace2（36 GB+）或更大的 WebFace42M/Glint360K。
- **GPU 决定 batch size**：论文 batch 1,800 对显存压力很大，消费级显卡需要 gradient accumulation 或多卡；小模型/小输入可单卡训练。

---

## 2. 数据集推荐组合

### 2.1 最小可行方案（MVP）

| 数据集 | 规模 | 用途 | 占用 |
|--------|------|------|------|
| CASIA-WebFace 112×112 | 491K 图 / 10.6K 人 | 训练 | ~2.7 GB |
| LFW | 13,233 图 / 5,749 人 | 评测 | ~180 MB |
| **合计** | | | **~3 GB** |

- **适合**：快速验证代码、跑通训练-评测 pipeline、教学演示。
- **预期结果**：LFW 95%–98%（小规模难以接近论文指标）。
- **时间**：NN4（96×96）单卡 1–2 天可跑完。

### 2.2 标准复现方案（推荐）

| 数据集 | 规模 | 用途 | 占用 |
|--------|------|------|------|
| MS1MV2 112×112 | 5.8M 图 / 85.7K 人 | 训练 | ~16.5 GB（.rec）/ ~25 GB 解压 |
| LFW | 13,233 图 / 5,749 人 | 评测 | ~180 MB |
| YouTube Faces（可选） | 3,425 视频 / 1,595 人 | 评测 | ~10 GB（精简版） |
| **合计** | | | **~27–35 GB** |

- **适合**：真正复现 FaceNet 方法并给出有竞争力的公开 benchmark 结果。
- **预期结果**：LFW 99.0%–99.5%，YouTube Faces 93%–95%。
- **注意**：MS1MV2 是 MXNet `.rec` 格式，需要写解码/解包脚本转成 PyTorch Dataset。

### 2.3 论文级大规模方案

| 数据集 | 规模 | 用途 | 占用 |
|--------|------|------|------|
| WebFace42M 或 Glint360K | 17M–42M 图 / 360K–2M 人 | 训练 | 100+ GB |
| LFW + YouTube Faces + IJB-B/C | 多个评测集 | 评测 | ~30 GB+ |
| **合计** | | | **>150 GB** |

- **适合**：想训练接近工业级规模的模型、做数据量消融研究。
- **要求**：必须上云或配备大容量本地存储 + 多卡训练。
- **不建议**作为 FaceNet 论文复现的默认方案。

---

## 3. 机器配置建议

### 3.1 最低配置（MVP 方案）

| 组件 | 规格 | 说明 |
|------|------|------|
| GPU | RTX 3060 / RTX 4060 12GB | 可训练 NN4 / NNS1 级别小模型 |
| CPU | 8 核以上 | 数据预处理/评测用 |
| 内存 | 16 GB | 够用 |
| 硬盘 | 256 GB SSD | 预留 CASIA + LFW + 代码 |
| 网络 | 稳定宽带 | Kaggle/InsightFace 下载 |

### 3.2 推荐配置（标准复现方案）

| 组件 | 规格 | 说明 |
|------|------|------|
| GPU | **RTX 4090 24GB** 或 **A100 40GB/80GB** | 24GB 可训练 NN3/NN4；80GB 可尝试更大 batch |
| CPU | 16 核以上 | 多 worker 加载数据 |
| 内存 | **64 GB** | 大 batch online mining 需要较多内存缓存 embeddings |
| 硬盘 | **1 TB NVMe SSD** | MS1MV2 + LFW + 预处理后数据 + checkpoint |
| 网络 | 稳定宽带 | 下载 16–50 GB 数据集 |
| 多卡 | 可选 2× RTX 4090 / 4× A100 | 若要复现 batch 1,800 或训练 NN2/NN1 大模型 |

### 3.3 云端/服务器配置（论文级方案）

| 组件 | 规格 | 说明 |
|------|------|------|
| GPU | 4×/8× A100 80GB 或 H100 | 训练 NN1/NN2 + WebFace42M |
| CPU | 32 核+ | 匹配多卡数据加载 |
| 内存 | 256 GB+ | 大 batch mining + 数据缓存 |
| 存储 | 2 TB+ NVMe / 网络存储 | WebFace42M 数百 GB |
| 预算 | 按小时计费云服务或本地服务器 | 训练周期数天到数周 |

---

## 4. 按目标选择配置

| 你的目标 | 推荐训练集 | 推荐 GPU | 推荐硬盘 | 预算参考 |
|----------|------------|----------|----------|----------|
| 跑通代码、验证方法 | CASIA-WebFace + LFW | RTX 3060 12GB | 256 GB SSD | 低 |
| 复现 FaceNet 并拿到 LFW 99%+ | MS1MV2 + LFW | RTX 4090 24GB / A100 40GB | 1 TB SSD | 中 |
| 追求接近论文的大规模结果 | WebFace42M / Glint360K | 多卡 A100 80GB | 2 TB+ | 高 |
| 只做推理/部署 | 下载预训练权重即可 | CPU 也可 | 50 GB | 很低 |

---

## 5. 数据集下载要点

1. **MS1MV2 推荐来源**：
   - Kaggle：`ms1m-arcface-dataset`（16.54 GB `train.rec`）
   - InsightFace DataZoo：GitHub `deepinsight/insightface/recognition/_datasets_`
   - 均为学术/非商业许可

2. **CASIA-WebFace 推荐来源**：
   - Kaggle：`casia-webface`（2.73 GB `.rec`）
   - 已 112×112 对齐，可直接使用

3. **LFW**：
   - 官网：`http://vis-www.cs.umass.edu/lfw/`
   - 完全免费，180 MB

4. **YouTube Faces**（可选）：
   - 官网：`http://www.cs.tau.ac.il/~wolf/ytfaces/`
   - 完整版约 10–25 GB，可后补

---

## 6. 风险提醒

- **不要试图下载原 MS-Celeb-1M**：微软已因隐私问题下架，使用社区清洗版 MS1MV2/MS1MV3。
- **VGGFace2 需要注册**：且 36 GB 压缩包对原机磁盘压力过大，换新机后若空间充足再考虑。
- **WebFace42M 需要签 License Agreement**：需要机构邮箱和负责人签字。
- **batch size 1,800 是论文设定**：消费卡放不下时可用 gradient accumulation 或减小 batch，但 semi-hard mining 效果会打折扣。

---

## 7. 建议的下一步

1. **换新机后先确认磁盘空间 ≥ 500 GB**，优先选择 **24GB+ 显存** 的 GPU。
2. 先下载 **LFW + CASIA-WebFace**，半天内跑通训练-评测 pipeline。
3. 代码验证无误后，下载 **MS1MV2**，作为主力训练集追求 LFW 99%+。
4. 记录实验结果，对比论文 Table 3/4/5/6 做消融分析。
