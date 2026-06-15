# FaceNet 复现可用公开训练数据集调研

> 调研日期：2026-06-13  
> 本机磁盘剩余：约 33 GB（/System/Volumes/Data）

---

## 1. 快速结论

| 优先级 | 数据集 | 规模 | 压缩/占用 | 是否推荐 | 理由 |
|--------|--------|------|-----------|----------|------|
| **首选训练集** | **MS1MV2 / MS1MV3** | 5.2M–5.8M 图 / 85K–93K 人 | ~16.5 GB（.rec），解压后 ~20–30 GB | ✅ 强推 | 当前复现 FaceNet 最主流的公开训练集；已清洗、已对齐；LFW 可训到 99.6%+ |
| **最小可行** | **CASIA-WebFace** | 494K 图 / 10.6K 人 | ~2.7 GB（.rec）/ 原始 ~4 GB | ✅ 适合 MVP | 小、快、公开易得；但规模只够验证 pipeline，难接近论文指标 |
| **备选训练集** | **VGGFace2** | 3.31M 图 / 9.1K 人 | train.tar.gz ~36 GB，解压 ~50+ GB | ⚠️ 磁盘吃紧 | 质量好、标签干净；但本机 33 GB 装不下原始压缩包，需外置存储或子集 |
| **大规模** | **WebFace42M / WebFace4M** | 42M / 4M 图 | 数百 GB | ❌ 不推荐当前做 | 最大公开清洗集；但数据量、训练时间、磁盘均远超本机承受 |
| **评测集** | **LFW** | 13,233 图 / 5,749 人 | ~180 MB | ✅ 必备 | 论文主评测集，公开免费 |
| **评测集** | **YouTube Faces** | 3,425 视频 / 1,595 人 | ~10–25 GB | ⚠️ 可选 | 论文次评测集；磁盘紧可跳过或用精简版 |

**综合意见**：

- 在本机 **33 GB** 磁盘约束下，**最现实的选择是 MS1MV2 + LFW**。MS1MV2 的 `.rec` 约 16.5 GB，解压成图片约 20–30 GB，勉强能放下；训练一个 NN2/NN4 级别的模型可以做到 LFW 99.0%–99.5%。
- 如果想先快速验证代码和流程，用 **CASIA-WebFace + LFW**，1 天内可跑通，但 LFW 大概只能到 95%–98%。
- **VGGFace2、WebFace42M、Glint360K** 对当前环境来说过大，除非外挂硬盘或使用云端存储/训练。
- 论文原训练集（100M–200M 图 / ~8M 人）**不公开**，所以任何公开数据复现都无法完全达到论文的 99.63%；MS1MV2 是最接近的实用替代。

---

## 2. 各数据集详细情况

### 2.1 CASIA-WebFace

| 项目 | 内容 |
|------|------|
| 来源 | 中科院自动化所（CASIA），Yi et al. 2014 |
| 规模 | 494,414 张图，10,575 个身份 |
| 下载方式 | 1) 官网申请：http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html（可能较慢/需审批）<br>2) Kaggle `.rec` 版（2.73 GB）：https://www.kaggle.com/datasets/debarghamitraroy/casia-webface<br>3) 百度网盘/谷歌盘社区镜像 |
| 格式 | 原始：按身份文件夹的 JPG；Kaggle/InsightFace：MXNet `.rec` + `.idx` |
| 对齐状态 | 原始为“loosely cropped”，需要用 MTCNN 重新检测对齐；Kaggle 版为 112×112 已对齐 |
| 质量 | 标签噪声中等，图像分辨率/姿态多样性一般；小规模时代的代表数据集 |
| 占用 | 原始约 4 GB；Kaggle `.rec` 约 2.7 GB；解压成图片约 3–5 GB |
| 许可 | 学术/研究用途 |

**适用性**：

- ✅ 代码冒烟测试、快速超参扫描、教学复现。
- ❌ 规模太小，难以复现论文级 LFW 99.6%；Triplet Loss 在这种小数据上容易过拟合。

---

### 2.2 MS-Celeb-1M / MS1MV2 / MS1MV3

| 项目 | 内容 |
|------|------|
| 来源 | Microsoft Research 原始发布；InsightFace/ArcFace 团队清洗 |
| 规模 | 原始 MS-Celeb-1M：~10M 图 / 100K 人（已下架）<br>**MS1MV2**：~5.8M 图 / 85.7K 人<br>**MS1MV3**：~5.2M 图 / 93K 人 |
| 下载方式 | 1) InsightFace DataZoo（GitHub）：https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_<br>2) Kaggle MS1M-ArcFace 112×112：https://www.kaggle.com/datasets/yakhyokhuja/ms1m-arcface-dataset/versions/1（16.54 GB `train.rec`）<br>3) 百度网盘/谷歌盘社区镜像 |
| 格式 | MXNet `.rec` / `.idx`（最常见）；部分镜像为 JPG 文件夹 |
| 对齐状态 | 已用 MTCNN/RetinaFace 5 点 landmark 对齐到 112×112 |
| 质量 | 经 ArcFace 团队清洗，标签噪声显著降低；是目前学术界训练人脸识别模型的“标准配置” |
| 占用 | `.rec` 约 16.5 GB；解压成 JPG 约 20–30 GB；本机 33 GB 可勉强容纳 |
| 许可 | 学术研究/非商业 |

**适用性**：

- ✅ **复现 FaceNet 的最佳公开训练集**：数量级足够大，能支撑 Triplet Loss 收敛到较高精度。
- ✅ 已对齐，可直接输入网络，减少预处理工作。
- ⚠️ 需要写 `.rec` 解码脚本或用 InsightFace 提供的 unpack 工具转换成 PyTorch Dataset。
- ⚠️ 原 MS-Celeb-1M 因隐私问题被微软下架，**不要试图从官网下载原始版**；用社区清洗版 MS1MV2/MS1MV3。

**预期结果**：

- 在 MS1MV2 上训练 ResNet50/Inception 级别的模型，LFW 通常可达 **99.5%–99.7%**，已经非常接近 FaceNet 论文的 99.63%。

---

### 2.3 VGGFace2

| 项目 | 内容 |
|------|------|
| 来源 | Oxford VGG，Cao et al. 2018 |
| 规模 | 3.31M 图 / 9,131 人；train 8,631 人，test 500 人 |
| 下载方式 | 官网注册：http://zeus.robots.ox.ac.uk/vgg_face2/<br>训练集：`vggface2_train.tar.gz` ~36 GB；测试集：`vggface2_test.tar.gz` ~1.9 GB |
| 格式 | JPG 文件夹 + `bb_landmark.tar.gz`（人脸框和 5 点 landmark） |
| 对齐状态 | “loosely cropped”，需要用 MTCNN 重新对齐裁剪 |
| 质量 | 标签较干净，姿态、年龄、人种多样性好 |
| 占用 | 压缩包 ~38 GB；解压后约 50–60 GB |
| 许可 | 学术研究 |

**适用性**：

- ✅ 数据质量高，常用来训练 face verification backbone。
- ❌ **本机 33 GB 磁盘放不下完整 VGGFace2**，需要外接硬盘或云端。
- ❌ 需要自己跑 MTCNN 做 5 点对齐，预处理耗时较长。
- ⚠️ 如果只想体验或做消融，可只下载 train 的一个子集。

---

### 2.4 WebFace260M / WebFace42M / WebFace4M

| 项目 | 内容 |
|------|------|
| 来源 | Tsinghua / XForwardAI / Imperial College，Zhu et al. 2021 |
| 规模 | WebFace260M：260M 图 / 4M 人（噪声大）<br>WebFace42M：42M 图 / 2M 人（清洗后）<br>WebFace4M：~4M 图 / 200K 人（子集） |
| 下载方式 | 官网申请 + 签署 License Agreement：https://www.face-benchmark.org/ |
| 格式 | JPG 文件夹；260M 仅提供图片 URL 列表（约 50 TB） |
| 对齐状态 | 已清洗，部分版本已对齐 |
| 质量 | 当前最大公开清洗人脸训练集；与工业级数据最接近 |
| 占用 | WebFace42M 数百 GB；WebFace4M 数十 GB |
| 许可 | 学术研究，需签署协议 |

**适用性**：

- ✅ 如果你想训练一个真正接近/超过 FaceNet 原数据规模的模型，这是公开选择。
- ❌ **远超本机磁盘和算力**：光是下载和存储就需要专门服务器。
- ❌ 申请流程较正式，需要机构邮箱/负责人签字。

---

### 2.5 Glint360K

| 项目 | 内容 |
|------|------|
| 来源 | InsightFace 团队，2021 |
| 规模 | 17.09M 图 / 360,232 人 |
| 下载方式 | 百度网盘 + Magnet URI；见 InsightFace GitHub |
| 格式 | MXNet `.rec` 多分卷 |
| 对齐状态 | 已对齐 112×112 |
| 质量 | 非常干净，大规模人脸识别 SOTA 常用 |
| 占用 | 解压后约 100+ GB |
| 许可 | 非商业研究 |

**适用性**：

- ✅ 大规模复现 ArcFace/AdaFace 等工作的首选。
- ❌ 对 FaceNet 论文复现来说规模过剩，且磁盘/训练成本过高。

---

### 2.6 LFW（评测集）

| 项目 | 内容 |
|------|------|
| 来源 | UMass Amherst，Huang et al. 2007 |
| 规模 | 13,233 图 / 5,749 人 |
| 下载方式 | http://vis-www.cs.umass.edu/lfw/ |
| 格式 | 按身份文件夹的 250×250 JPG |
| 占用 | ~180 MB |
| 许可 | 公开免费 |

**适用性**：

- ✅ 论文主评测集，必须下载。
- ✅ 提供 original / funneled / LFW-a / deep funneled 多个版本；FaceNet 论文使用中心裁剪或额外对齐两种模式评测。

---

### 2.7 YouTube Faces DB（评测集）

| 项目 | 内容 |
|------|------|
| 来源 | Tel Aviv University，Wolf et al. 2011 |
| 规模 | 3,425 视频 / 1,595 人 |
| 下载方式 | http://www.cs.tau.ac.il/~wolf/ytfaces/ |
| 格式 | 视频文件 + `.mat` 元数据（人脸框、姿态角） |
| 占用 | 完整视频约 10–25 GB；OpenDataLab 镜像标 24.5 GB |
| 许可 | 学术研究 |

**适用性**：

- ✅ 论文次评测集（95.12% accuracy）。
- ⚠️ 评测时需要从视频中抽帧、检测人脸、计算平均相似度，流程比 LFW 复杂。
- ❌ 磁盘紧张时可跳过，或只用精简版（如 YouTube Faces with Facial Keypoints，~10 GB）。

---

## 3. 针对本机环境的建议组合

### 方案 A：最小可行路线（MVP，推荐先跑通）

| 数据集 | 用途 | 预计占用 |
|--------|------|----------|
| CASIA-WebFace 112×112 | 训练 | ~2.7 GB |
| LFW | 评测 | ~180 MB |
| **合计** | | **~3 GB** |

- 1–2 天可训练 NN4（96×96）或小模型。
- 目标：LFW 95%–98%，验证 Triplet Loss + online mining 代码正确。

### 方案 B：标准复现路线（最接近论文）

| 数据集 | 用途 | 预计占用 |
|--------|------|----------|
| MS1MV2 112×112 | 训练 | ~16.5 GB（.rec）/ ~25 GB（解压） |
| LFW | 评测 | ~180 MB |
| YouTube Faces（可选） | 评测 | ~10 GB（精简版） |
| **合计** | | **~27–35 GB** |

- 需要精打细算磁盘空间：训练时尽量直接读取 `.rec` 或在 `.rec` 上生成 HDF5/lmdb，避免同时保留 `.rec` + 解压图两份。
- 目标：LFW 99.0%–99.5%，YouTube Faces 93%–95%。

### 方案 C：论文级大规模（不推荐本机执行）

| 数据集 | 用途 | 预计占用 |
|--------|------|----------|
| WebFace42M 或 Glint360K | 训练 | 100+ GB |
| LFW + YouTube Faces + IJB-C | 评测 | ~30 GB |
| **合计** | | **>150 GB** |

- 需要外接硬盘或云端训练（AWS/GCP/阿里云）。
- 即便如此，仍无法复现 Google 内部 100M–200M 训练集的 exact 结果。

---

## 4. 下载与格式转换注意事项

1. **Kaggle 数据集**：
   - 需要 Kaggle 账号 + API token（`kaggle.json`）。
   - 命令示例：
     ```bash
     kaggle datasets download -d debarghamitraroy/casia-webface
     kaggle datasets download -d yakhyokhuja/ms1m-arcface-dataset
     ```
   - 下载的是 `.rec`/`.idx`，需要转换成 PyTorch 可读的格式。

2. **InsightFace DataZoo**：
   - GitHub 仓库提供百度网盘和 Google Drive 链接。
   - 包含 `unpack_*.py` 脚本，可把 `.rec` 解包成图片。

3. **MXNet `.rec` 转 PyTorch**：
   - 可用 `mxnet.recordio.MXIndexedRecordIO` 读取。
   - 或先 unpack 成 `identity_folder/image.jpg`，再写 `ImageFolder`。
   - 也可以直接写自定义 `Dataset` 从 `.rec` 按需读取，避免解压占用双倍空间。

4. **预处理对齐**：
   - CASIA-WebFace Kaggle 版、MS1MV2 都已经 112×112 对齐，可直接 resize 到 96/160/224。
   - VGGFace2 原始版需要用 MTCNN 检测并裁剪，会额外消耗时间/空间。

---

## 5. 风险提示

| 风险 | 说明 |
|------|------|
| 原训练集不公开 | 论文的 100M–200M Google 内部数据拿不到，公开数据复现 LFW 上限大约在 99.5%–99.7% |
| 数据集许可 | MS1MV2、VGGFace2、WebFace 多为学术/非商业许可；商用需谨慎 |
| 隐私/合规 | MS-Celeb-1M 原始版已下架，使用社区清洗版时避免再分发 |
| 磁盘空间 | 本机仅 33 GB；MS1MV2 + LFW 是上限，VGGFace2/WebFace 必须外挂存储 |
| 标签噪声 | CASIA-WebFace 噪声相对高；建议用 MS1MV2 作为主力训练集 |

---

## 6. 最终推荐

> **如果你的目标是“在本机快速、可落地地复现 FaceNet 论文方法并给出可对比的 LFW 结果”**：
>
> 1. **立即下载 LFW**（180 MB）。
> 2. **下载 MS1MV2 112×112**（Kaggle 16.5 GB `.rec`），作为主力训练集。
> 3. 先写 `.rec` → PyTorch 的读取器，确认能加载。
> 4. 如果磁盘吃紧或想更快验证，先用 **CASIA-WebFace 112×112** 跑通训练-评测 pipeline。
> 5. **YouTube Faces 可选**，有空间再下；优先保证 LFW 99%+ 的结果。
