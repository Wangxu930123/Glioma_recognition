# 赛道4 · 基于 MRI 的脑胶质瘤智能检测（自建模型组 · 算法工程）

> 本仓库是"自建模型组"的 **训练 + 推理** 工程：从 clone 代码 → 装环境 → 搬数据 →
> 跑通全链路 → 训模型 → 评估 → 导出提交权重。对齐《赛事开发规范（赛道四）V1.0》
> 与《公共数据集格式说明》。
>
> **要一步步照抄命令、看期望输出、查报错** →
> 用 [`docs/TRACK4_RUNBOOK.md`](docs/TRACK4_RUNBOOK.md)（探针 → 折划分 → 缓存 → 训练 →
> 收尾 → 导出 → 提交，每条命令都可直接复制粘贴，含验收判据与报错对照）。
> 本文是**参考手册**（配置项含义、算法细节、指标定义、为什么这么设计）。

### 点这里（按"你要做的事 / 卡在哪"查）

| 你要做的事 / 卡在哪 | 点这里 |
|---|---|
| **照着复制粘贴跑完整流程**（期望输出、验收判据、报错对照） | [`docs/TRACK4_RUNBOOK.md`](docs/TRACK4_RUNBOOK.md) —— §0.1 一页照抄版；§0 三条铁律 |
| 交付什么、评分点在哪 | 本文 §0 |
| 装环境 / 依赖（联网 / 离线 / 容器） | 本文 §1；镜像与离线细节 [`docs/CLOUD_DESKTOP_RUNBOOK.md`](docs/CLOUD_DESKTOP_RUNBOOK.md) |
| 数据搬迁、数据根怎么填 | 本文 §2；单点排查 [`docs/DATASET_ROOT_TROUBLESHOOT.md`](docs/DATASET_ROOT_TROUBLESHOOT.md) |
| 全流程命令速查 | 本文 §3（逐节指向手册） |
| 要不要交叉验证 / 能不能全量训练 + 验证集 | 本文 §4；**验证集没有金标准怎么办** → 手册 §8 |
| 评估口径（external / 折内留一、OOF） | 本文 §5 |
| 配置项含义（paths / train / preprocess） | 本文 §6 |
| 报错怎么修 | 手册 §16 报错对照表；本文 §7 常见问题 |
| 提交前检查项、红线 | 本文 §8.2 / §8.3；手册 §12 / §11 |
| 平台怎么点、账号、建容器实例 | [`docs/PLATFORM_GUIDE.md`](docs/PLATFORM_GUIDE.md) |
| 三工程怎么串、容器内全链路 | [`docs/MASTER_GUIDE.md`](docs/MASTER_GUIDE.md)、[`docs/CLOUD_DESKTOP_RUNBOOK.md`](docs/CLOUD_DESKTOP_RUNBOOK.md) |

## 目录

| 章节 | 内容 |
|---|---|
| [0](#0-任务与交付物) | 任务与交付物（交什么、评分点在哪） |
| [1](#1-获取代码与环境) | 获取代码、依赖安装（联网/离线/平台容器）、目录结构 |
| [2](#2-数据准备与搬迁) | 数据布局、从别处搬迁数据、设置数据根、数据体检 |
| [3](#3-全流程从空目录到提交权重) | 全流程速查（一行命令 + 逐节指向照抄手册） |
| [4](#4-训练策略不做交叉验证行不行) | **要不要交叉验证 / 能不能全量训练+验证集** |
| [5](#5-评估口径external-与折内留一oof) | 评估口径：external 与折内（留一/OOF） |
| [6](#6-配置速查) | 配置速查（paths / train / preprocess） |
| [7](#7-常见问题) | 常见问题 |
| [8](#8-附录) | 附录：脚本清单、自检清单、红线 |
| [照抄手册](docs/TRACK4_RUNBOOK.md) | **一步步复制粘贴**：变量表、每步期望输出、验收判据、报错对照、提交前清单 |

---

## 0. 任务与交付物

### 0.1 评分点 → 本工程实现

| 目标 | 规范要求 | 本工程实现 |
|---|---|---|
| **目标一** 假人体 | `prediction.json.IsNotHumanBodyProb` | `special` 头（`annotation/fake` 监督）+ 保守启发式兜底 |
| **目标二** 拼接影像 | `prediction.json.IsStitchedProb` | `special` 头（`annotation/Composition` 监督）+ 启发式兜底 |
| **重复影像** | `duplicate_pairs.jsonl`：`{StudyUID, StudyUID_dup, PairProb}`，每例 ≤200 对、缺省 0；AUC-PR / Recall@10%FPR / Precision@15%Recall | 学习式嵌入（金标准正/负对训练）+ 几何·强度指纹融合 → 阈值标定 → 每例 Top-200 |
| **任务A** | **T1 增强核心区**二值掩码；仿射与维度**必须与 T1C 原图一致**；体素严格 0/1 | 滑窗 → 阈值 → 连通域后处理 → 回采样到 T1C 原始空间 → 写盘后回读复核 |
| **任务B** | **FLAIR/T2 总异常区**二值掩码；仿射与维度**必须与 FLAIR/T2 原图一致** | 同上（回采样到 FLAIR/T2 原始空间），并强制 `peri ⊇ core` |
| **结构化诊断** | `Prediction` 14 字段 + `Interpretation.Conclusion` | 多任务头（逐字段 `label_mask` 稀疏监督）+ 规则校正 |
| 日志 | JSONL，路径**必须** `/2026aicompetition/workspace/logs`，字段 `timestamp/epoch/step/phase/mode/loss/lr/data_source/checkpoint/pretrained_from` | `src/utils/logger.py`（平台目录自动切换） |
| 推理服务 | `POST /call`（嵌套 `input:{evaluation_id,dataset_path}`）、5s 内回 200、`GET /health` 200、结果写 `answer/{evaluation_id}/`、回调平台 | `src/serving/app.py`（后台异步 + 启动预检 + 回调重试 + `/status`） |
| 提交格式 | 掩码体素严格 0/1、仿射/维度与对应模态一致，否则该例分割直接 0 分 | `write_mask` 写盘后回读复核 + `validate_answer` 复检 |

### 0.2 交付物

1. **提交权重**：按规范 §5.2 放到 `/2026aicompetition/workspace/checkpoint/{goal1_authenticity,goal2_stitched,goal2_duplicate,goal3_tumor,goal4_diagnosis,goal5_segmentation}/`，由 `bash scripts/09_export_submission.sh` 完成（多折时 `goal5_segmentation/` 保留全部折，评测容器自动集成）。
2. **训练日志**：`/2026aicompetition/workspace/logs/*.jsonl`。
3. （团队串联模式）插件：`integration/tasks.py` 的 `StudyTask` / `DatasetTask`。

---

## 1. 获取代码与环境

### 1.1 获取代码

```bash
git clone <仓库地址> glioma_track4
cd glioma_track4
```

代码目录可放任意位置（如 `~/glioma_track4`、`/2026aicompetition/workspace/glioma_track4`）。
下文所有命令假设**当前目录就是仓库根**（脚本内部会自动切到仓库根）。

### 1.2 安装依赖

```bash
# ① 本地/云主机（有网，默认 conda 建独立环境）
bash scripts/00_setup_env.sh

# ② 训推平台容器（镜像自带 torch，只补其余包）
bash scripts/00_setup_env.sh --mode system

# ③ 无网机器（云桌面）：先在联网机下 wheel，再离线装
bash scripts/00b_prepare_wheels.sh --with-torch --mirror tsinghua --out wheels
#    把 wheels/ 拷到无网机后：
bash scripts/00_setup_env.sh --mode system --offline --wheels wheels

# ④ 只体检不安装（列出缺哪些包、torch/cuda 是否可用）
bash scripts/00_setup_env.sh --mode verify
```

常用参数：`--mirror tsinghua`（换源）、`--torch-index mirror`、`--name <env>`（conda 环境名）、`--python <解释器>`、`--with-torch`。

### 1.3 解释器与依赖

- **Python ≥ 3.10，推荐 3.11**（代码用到 `int | None` 等新语法）。
- 容器里常常只有 `python3`、没有 `python`：shell 脚本会自动探测 `python3` → `python`，也可 `export PY=/opt/miniconda3/envs/xxx/bin/python`。
- 关键依赖：`torch>=2.4`（平台镜像自带）、`numpy`、`scipy`、`nibabel`、`SimpleITK`、`scikit-image`、`pandas`、`openpyxl`、`pyyaml`、`fastapi`、`uvicorn`、`pydantic`、`requests`（见 `requirements.txt`）。
- 显存：`batch_size=2`、`patch=96³`、`bf16` 约 16GB 起步；不足见 [7.3](#73-显存不足)。

### 1.4 目录结构

```text
glioma_track4/
├── configs/                 paths / preprocess / train / train_large / labels
├── src/
│   ├── utils/               配置加载（环境变量覆盖）+ 规范 JSONL 日志
│   ├── data/                probe 探针、dicom、dataset（网格/增广/多折）、labels
│   ├── models/              mednext（主骨干）+ unet3d（备选/多任务头）
│   ├── training/            损失 + 指标 + 训练器（AMP/EMA/断点/四路损失）
│   ├── evaluation/          离线评估（Dice/NSD/HD95）、划分解析
│   ├── inference/           滑窗+TTA+集成、病例流水线、结构化推导、重复匹配、答案写出
│   └── serving/             FastAPI 服务（/call + /health + 回调）
├── integration/             团队串联提交侧插件（export_ckpt / tasks / common）
├── scripts/                 全流程脚本（见 8.1）
├── data/                    清单、折划分、缓存、统计（探针产出）
├── checkpoints/             训练产物 g4_fold*/best.pth
└── logs/                    训练/评估/服务日志
```

---

## 2. 数据准备与搬迁

### 2.1 需要哪些数据

| 数据 | 是否必需 | 用途 |
|---|---|---|
| **训练集**（`training/`） | 必需 | 训练 + 折内验证 |
| **验证集**（`verification/`） | 强烈建议 | 最终评估口径（与训练集无交集，无需留一）；也可作"全量训练"的早停集。**实测只有 `SeriesType.xlsx`、没有字段金标准表**，见 §6.1 |
| 评测集（`evaluation_*`） | 不必准备 | 评测时由平台通过 `input.dataset_path` 下发 |

平台挂载下每个阶段是一个平行目录：

```text
/2026aicompetition/datasets/
├── training/            ← 官方训练集（数据根取这一层）
├── verification/        ← 验证集（建议接入）
├── evaluation_first/    ← 评测输入（推理用）
├── evaluation_second/
└── evaluation_finals/
```

### 2.2 数据布局（数据路径 / 数据信息路径）

以训练集为例，**阶段目录**下是 `annotation/`：

```text
<阶段>/annotation/
├── SeriesType.xlsx                                  ★ 数据信息路径（模态表）
├── 脑胶质瘤标注结果-训练集.xlsx                       ★ 数据信息路径（结构化字段金标准，训练集才有）
├── <32位检查号>/                                    ★ 数据路径（影像）：每个检查号一个病例目录
│   └── <2.25.* 序列UID>/<序列UID>.nii.gz
└── {fake, compositing, duplicate}/                  特殊影像 / 重复影像金标准
```

| 项 | 位置 |
|---|---|
| **影像数据** | `<阶段>/annotation/<32位检查号>/<2.25.* 序列UID>/<序列UID>.nii.gz` |
| **数据信息（序列类型）** | `<阶段>/annotation/SeriesType.xlsx`（**与病例目录同层**） |
| 列 / 取值 | `AccessionNumber` + `SeriesUid` + `SeriesType` ∈ {`T1`, `T1CE（增强）`, `T2-Flair`, `T2WI`, `其他`} |
| 取值 → 通道 | `T1`→t1、`T1CE（增强）`→t1c、`T2-Flair`→flair、`T2WI`→t2、`其他`→**排除** |

要点：

- **数据根填阶段目录即可**：`DATASET_ROOT=/2026aicompetition/datasets/training`（代码会自动下钻到 `annotation/`，也可直接填 `.../training/annotation`）。填 `/2026aicompetition/datasets` 这类**多阶段父目录**会被 `ValueError` 拦住——停在上层不会当场报错，而是把阶段名当检查号、清单里出现假病例、金标准一张都对不上。
- `annotation/{fake,Composition,duplicate}` **不是**检查号目录，扫描时跳过；其中病例只作为**特殊影像正样本**补进清单。
- 序列目录/文件名是 DICOM UID，**靠关键词猜不出模态**，必须靠 `SeriesType.xlsx`；探针按候选目录搜该表（`$GLIOMA_LABELS_DIR` → `<工程>/labels` → `$WORKSPACE` 下 3 层 → 数据根/父/祖父），所以数据根填哪一层都能命中。
- `SeriesType=其他` 是**权威排除**，探针直接跳过（报告里的 `cases_with_declared_other_series` 是它的计数，非 0 属正常）。
- **`3_serieslabel.xlsx` 不是本赛道数据集的内容**（属工作区里另一个目标的产物），**与赛道四数据集没有关系**：它取值粗（只写 `T2`），混用会把 `T2WI`/`T2-Flair` 静默覆盖成 `T2`。模态只认数据集自带的 `SeriesType.xlsx`。
- 字段金标准 `annotation/脑胶质瘤标注结果-训练集.xlsx` 有 **3 张工作表**（`检查级别`/`序列级别`/`ROI级别`）：**病例级字段只取 `检查级别`**；序列级/ROI 级的行按病例嵌套保留（`__series_rows__`/`__roi_rows__`）但**不写进 `manifest.json`**（一例多行，按检查号硬合并会覆盖病例级字段且不报错）。
  **唯一例外**：某病例在 `检查级别` 里没有行、只出现在序列级/ROI 级表里时，若该病例**所有子行的标签列**（`病理结果` / `WHO分级` / `Glioma`）取值**完全一致**，就把这一个值补进病例级；不一致则告警且**不采纳**。只放开标签列是因为只有丢它们会静默缩小评测分母——整行兜底会**任取一行**（实测同一病例两条序列行的 `Signal_T2WI` 分别是 `High`/`Low`），比读不到更难查。
- `检查级别` 的**实测排版**（别按"第一行就是表头"去读）：**表头在第 2 行**（第 1 行是标题），第 **H** 列 `StudyUid`、第 **I** 列 `Accessionumber`（少一个 n 的拼写，行键取它、**不是** `StudyUid`），第 **N** 列 `Study->CLINICAL->病理结果`（列名是**字段路径**，末段才是字段名）。取值形如 `脑胶质瘤2级` / `脑胶质瘤4级` / `其他肿瘤或病变` / `病因不明`。
  代码对这三点的处理：表头在前 20 行里扫描（不写死行号）；`accessionnumber` 的拼写变体与 `->`/`→` 分层列名都按**末段**匹配（拼错或分层都不会再串列）。
- `ROI级别` 的**实测排版**：表头同样在**第 2 行**，第 **H** 列 `StudyUid`、第 **I** 列 `AccessioNumber`（行键取它）、第 **Y** 列 `SerisDescription`（`T1` / `T2-FLAIR` / `T1CE（增强）` / `其他`）、第 **AB** 列 `ROIUid`、第 **AC** 列 `RoiName`（`瘤体` / `水肿` / `肿瘤瘤体` / `全肿瘤`）、第 **AQ** 列 `Study->CLINICAL->病理结果`（同上，用于病例只在这张表里时兜底）。
  行键必须落在 `RoiName` 上：只写 `roi` 前缀的话会先撞上位置更靠前的 AB 列 `ROIUid`，ROI 名（**掩膜角色 core/peri 的唯一来源**）就退化成普通列了。
- 级别/病名支持多种写法：列名 `WHO_grade` 或 `WHO分级`，取值 `4` / `4.0` / `4级` / `Ⅳ` / `IV` 都认；`病理结果` 写 `胶质瘤`（不带级数）也算阳性；`其他肿瘤或病变` / `病因不明` 这类**非胶质瘤取值映射为 `TumorProbability=0`**（不是"没有金标准"——否则指标分母会悄悄变小）。
- 看某张表的真实列名与行数：`python scripts/30_inspect_table.py <表文件> --rows 3`（表放哪都行，离线也能看；表在**数据根上一级**时探针同样能扫到）。

### 2.3 搬迁数据

```bash
# ① 整体搬迁（推荐：rsync 可断点续传、保留结构）
rsync -av --info=progress2 /源路径/training/ /目标路径/datasets/training/

# ② 只搬一部分病例（本地小规模实验）
rsync -av /源/training/annotation/SeriesType.xlsx /目标/datasets/training/annotation/
rsync -av /源/training/annotation/<32位检查号>/ /目标/datasets/training/annotation/<32位检查号>/

# ③ 数据不动，指过去（软链：不占空间，适合只读大盘）
mkdir -p /目标/datasets && ln -s /源路径/training /目标/datasets/training
```

**搬迁检查清单**（搬完立刻验，别等训练完才发现缺序列）：

- [ ] `<阶段>/annotation/SeriesType.xlsx` 存在（**必须与病例目录同层**；漏搬的表现是"病例数正常、却报无任何可用序列"）
- [ ] 病例目录数与源一致：`ls <阶段>/annotation | grep -E '^[0-9a-f]{32}$' | wc -l`
- [ ] 每个病例目录下有序列子目录，且 `<序列UID>.nii.gz` 与目录名一致
- [ ] 训练集额外有 `脑胶质瘤标注结果-训练集.xlsx`（结构化字段监督用）
- [ ] 磁盘空间足够（原始 NIfTI 数百 GB，预处理缓存另算）

### 2.4 设置数据根与环境变量

优先级：**环境变量 > `configs/paths.yaml`**。临时用 export，长期改 yaml。

```bash
export DATASET_ROOT=/2026aicompetition/datasets/training   # 训练集根（也可填 .../training/annotation）
export VAL_ROOT=/2026aicompetition/datasets/verification   # 验证集根（可选，强烈建议）
export CACHE_DIR=/2026aicompetition/workspace/cache        # 预处理缓存（放持久化目录）
export WORKSPACE=/2026aicompetition/workspace              # 日志/答案/权重根
export PY=/opt/miniconda3/envs/gl_py311/bin/python         # 需要时显式指定解释器
```

| 变量 | 作用 | 默认 |
|---|---|---|
| `DATASET_ROOT` | 训练集数据根 | `paths.yaml: raw.track4` |
| `VAL_ROOT` | 验证集数据根 | `raw.val`（默认空 = 不启用） |
| `MANIFEST` / `VAL_MANIFEST` | 直接指定清单文件 | `data/manifest.json` / `data/manifest_val.json` |
| `CACHE_DIR` | 预处理缓存目录 | `data/preprocess_cache` |
| `CKPT_DIR` | 权重目录 | `checkpoints` |
| `LOGS_DIR` | 日志目录 | `logs`（平台环境自动切 `{WORKSPACE}/logs`） |
| `WORKSPACE` | 平台工作区根 | `/2026aicompetition/workspace` |
| `ANSWER_ROOT` | 推理结果目录 | `{WORKSPACE}/answer` |
| `GLIOMA_CHECKPOINT_ROOT` | 提交权重导出目录 | `{WORKSPACE}/checkpoint` |
| `GLIOMA_CKPT` | 推理/服务权重（逗号分隔 = 集成） | 自动发现 |
| `GLIOMA_LABELS_DIR` | 显式指定数据信息表目录 | 自动搜索 |
| `GLIOMA_LOADER_TOLERANT` | 推理加载容错（1=坏文件跳过） | `0`（`06_platform_serve.sh` 里默认开 1） |
| `PY` | Python 解释器 | 自动探测 `python3`→`python` |

### 2.5 数据体检

```bash
python scripts/29_locate_dataset_root.py --base /2026aicompetition   # 找不到数据根时
python scripts/30_inspect_table.py --root $DATASET_ROOT --rows 3     # 摊开数据信息表
bash scripts/01_probe.sh                                             # 正式探针 → data/manifest.json
```

探针报告要看的 5 个字段：

| 字段 | 期望 | 异常含义 |
|---|---|---|
| `case_count` | 与源一致 | 少 → 数据根填错/漏搬 |
| `series_type_rows` | > 0 | 0 → 没找到 `SeriesType.xlsx`（设 `GLIOMA_LABELS_DIR`） |
| `modality_counts` | 出现 t1c/flair/t2/t1 | 全 `other` → 模态表没命中 |
| `missing_t1c` / `missing_flair` | 少量正常 | 大量 → 数据不全（会自动降级，但掉点） |
| `special.gold_pairs` | > 0 | 0 → 重复影像金标准格式不匹配（目标二失去监督） |

标签表异常（`Label（来源）` 列写错等）：

```bash
bash scripts/33_fix_abnormal_labels.sh              # 全自动：定位 → 核验 → 合并 → 重跑探针
bash scripts/33_fix_abnormal_labels.sh --dry-run    # 只看不写
python scripts/32_apply_abnormal_patch.py --patch <修补版.xlsx>   # 手动核验/合并（默认只报告）
```

### 2.6 本地验证数据 vs 官方数据（数据源卫生）

清单里的 `data_source` 按**实际数据根**自动记录 `official/*` 或 `local/*`；`assert_data_source` 闸门**拒绝**"清单与本机数据根不同类"的训练与评估，防止把本地小数据的权重当正式结果。

```bash
# 从本地实验切到官方数据（顺序很重要）
bash scripts/17_reset_for_official.sh                     # ① 归档本地产物（不删除）
export DATASET_ROOT=/2026aicompetition/datasets/training  # ② 官方数据根
export CACHE_DIR=/2026aicompetition/workspace/cache       # ③ 缓存放持久化目录
bash scripts/01_probe.sh && bash scripts/02_build_dataset.sh
python scripts/13_build_cache.py --workers 8
```

---

## 3. 全流程（从空目录到提交权重）

### 3.1 步骤速查（命令 + 指向照抄手册）

> **命令细节、期望输出、验收判据、报错对照全部在
> [`docs/TRACK4_RUNBOOK.md`](docs/TRACK4_RUNBOOK.md)**（可直接复制粘贴），本节只做索引，两边不会不同步。
> 变量先一次性配好（`PY`/`PYTHON`/`WORKSPACE`/`DATASET_ROOT`/`VAL_ROOT`/`CACHE_DIR`/`GLIOMA_CHECKPOINT_ROOT`）→ 手册 §1。

| 步 | 做什么 | 一行命令 | 产物 / 判据 | 照抄版 |
|---|---|---|---|---|
| 0 | 环境体检 + 冒烟 | `bash scripts/00_setup_env.sh --mode verify`；`python scripts/99_smoke_test.py` | `PASS ✔` | [手册 §2](docs/TRACK4_RUNBOOK.md) |
| 1 | 定位数据根 + 探针 | `python scripts/29_locate_dataset_root.py`；`bash scripts/01_probe.sh`（验证集加 `--val`） | `data/manifest.json`、`manifest_val.json` | [手册 §3](docs/TRACK4_RUNBOOK.md)、[§4](docs/TRACK4_RUNBOOK.md) |
| 2 | 折划分 + 互验 | `bash scripts/02_build_dataset.sh`；`python scripts/24_verify_eval_split.py` | `data/folds.json`；**0 WARN** | [手册 §5](docs/TRACK4_RUNBOOK.md) |
| 3 | 预处理缓存 | `python scripts/13_build_cache.py --workers 8` | 缓存目录；二次运行秒级 | [手册 §6](docs/TRACK4_RUNBOOK.md) |
| 4 | （可选）性能基准 | `python scripts/12_bench.py --patch 96 --batch 2 --steps 12` | 定 patch/batch/折数 | [手册 §10](docs/TRACK4_RUNBOOK.md) |
| 5 | 训练 | `bash scripts/03_train.sh all 4`，随后 `bash scripts/03_train.sh 4`（第 5 折）；全量：`bash scripts/03_train.sh full` | `checkpoints/g4_fold*/best.pth` | [手册 §7](docs/TRACK4_RUNBOOK.md)、[§8](docs/TRACK4_RUNBOOK.md) |
| 6 | 收尾一条龙 | `FOLDS="0 1 2 3 4" bash scripts/16_finalize.sh` | 标定阈值 + 评估指标 + 导出权重 | [手册 §9](docs/TRACK4_RUNBOOK.md) |
| 7 | 导出 + 提交前检查 | `bash scripts/09_export_submission.sh`；`bash scripts/23_pre_submit_check.sh` | `{WORKSPACE}/checkpoint/goal5_segmentation/`；**0 失败** | [手册 §11](docs/TRACK4_RUNBOOK.md)、[§12](docs/TRACK4_RUNBOOK.md) |
| 8 | 协议闭环 + 服务 | `bash scripts/08_mock_competition.sh`；`CALLBACK_URL=... bash scripts/06_platform_serve.sh` | 五项 `PASS` | [手册 §13](docs/TRACK4_RUNBOOK.md) |

三个最容易踩的坑（手册里有完整版）：**①先探针再建折**（换数据根后旧折静默失效）；
**②演练权重隔离**（`TAG_PREFIX=smoke`）；**③`PY` 与 `PYTHON` 两个变量都要设**（`01`~`05` 读 `PY`，`16`/`09`/`23`/`08` 读 `PYTHON`）。

### 3.2 常用变体（按需加参数）

| 想干什么 | 命令 |
|---|---|
| 冒烟（无需真实数据） | `python scripts/99_smoke_test.py` |
| 探针：显式数据根 / 接验证集 | `bash scripts/01_probe.sh /path/to/training`；`export VAL_ROOT=... && bash scripts/01_probe.sh --val` |
| 指定折数 | `N_FOLDS=5 bash scripts/02_build_dataset.sh` |
| 缓存强制重建（数据变了） | `python scripts/13_build_cache.py --workers 8 --force` |
| 单折前台调试 | `bash scripts/03_train.sh 0` |
| 补第 5 折 | `bash scripts/03_train.sh 4`（等价 `FOLD=4 bash scripts/03_train.sh one`） |
| 换配置 / 加载初始化权重 | `CONFIG=train_large bash scripts/03_train.sh 0`；`PRETRAINED=/path/to.pth bash scripts/03_train.sh 0` |
| 只看当前评估口径（不训练） | `bash scripts/16_finalize.sh --print-split` |
| 分步收尾 | `python scripts/14_calibrate_thresholds.py --split external --limit 30 --write`；`CKPT=a.pth,b.pth bash scripts/04_eval.sh --split external`；`python scripts/15_eval_special_dup.py --split external` |
| 只校验导出结果 | `bash scripts/09_export_submission.sh --verify` |
| 快速提交前检查（跳过耗时项） | `bash scripts/23_pre_submit_check.sh --quick` |
| 平台起服务 / 本机自测 | `PORT=8000 CALLBACK_URL=http://<平台>/callback bash scripts/06_platform_serve.sh`；`CKPT=checkpoints/g4_fold0/best.pth bash scripts/05_serve.sh` |

> 每条命令的**期望输出、验收判据与失败处理**见 [`docs/TRACK4_RUNBOOK.md`](docs/TRACK4_RUNBOOK.md)；
> 报错对照表在手册 §16，提交前红线在手册 §11/§12。

### 3.3 每步在做什么（原理速览）

| 步 | 关键机制 |
|---|---|
| 1 探针 | 扫 `<阶段>/annotation` → 病例目录；靠 `SeriesType.xlsx` 定模态；`其他` 排除；读 3 张字段表（病例级字段取 `检查级别`，标签列可在子行取值一致时兜底）；扫 `fake/Composition/duplicate` 得特殊影像与重复对，写 `special` |
| 2 分折 | 按**分层键**（t1c/flair/t2 是否齐全 + 特殊影像标签）分层做 5 折，保证每折分布一致；同时生成 GoldPairs 训练对 |
| 3 缓存 | 把每例重采样到 1mm 公共网格（整脑 + patch 视图），训练时只读 `.npz`；**不缓存掩码的回采样结果**以外的任何非确定性内容 |
| 4 基准 | 实测 1 步耗时/峰值显存/推理单例耗时 → 反推 patch、batch、TTA 次数与可跑折数 |
| 5 训练 | 四路损失：分割（core/peri）+ 分类（fake/comp/重复对）+ 结构化字段（稀疏 `label_mask`）；AMP bf16 + EMA + 断点续训；每轮用**折内 val**（或全量模式下用**官方验证集**）算 Dice 存 `best.pth` |
| 6 评估 | 阈值标定（集成生效阈值一致）→ 全图推理（滑窗 + TTA + 集成）→ 分割指标；目标一二/重复影像按口径（external 或 OOF）评；收尾串联导出权重 |
| 7 导出 | 把权重按 §5.2 六目录铺开；`answer/` 用真实数据跑一遍协议闭环并复算指标 |
| 8 服务 | `/call` 5s 内回 200（后台异步），结果写 `answer/{evaluation_id}/`，完成后回调平台 |

---

## 4. 训练策略：不做交叉验证行不行

### 4.1 结论

- **规范没有强制交叉验证**。交叉验证是"**没有独立验证集**时"用来做模型选择/早停/泛化估计的手段。
- 有官方验证集（`verification/`，与训练集无交集）时，**"整个训练集训练 + 用验证集看指标"是完全合法的做法**，而且用满了 100% 数据。
- 但"是否需要 K 折"取决于你要的是哪一样：

| 你想要的 | 只有官方验证集够吗 | 还需要 K 折吗 |
|---|---|---|
| 一个能提交的模型 | 够（训练集全量 + 验证集早停） | 不需要 |
| **多模型集成**（通常 +1~3 分） | 不够（要多个模型） | 需要（或全量训多个 seed） |
| 无泄漏的泛化估计 | 够（验证集独立） | 不需要（OOF 也只在没有验证集时才必要） |
| 训练稳定性/方差检查 | 不够（验证集只有一份） | 需要 |

### 4.2 三种做法对比

| 做法 | 训练数据 | 早停/选模 | 交付 | 优点 | 代价 |
|---|---|---|---|---|---|
| **A. 5 折 + 全折集成**（现状默认） | 每折 80% | 折内 val（+ 验证集监控） | 5 个权重集成 | 集成增益最大、稳、可用 OOF 看无偏指标 | 5 倍训练时间 |
| **B. 全量单模型** | 100% | 官方验证集 | 1 个权重 | 数据最全、最省算力 | 无集成增益；早停与最终指标同源，指标略乐观 |
| **C. 全量多 seed**（折中） | 100% × N | 官方验证集 | N 个权重集成 | 数据全 + 有集成 | N 倍训练时间 |
| **D. 5 折 + 全量**（效果最好） | — | — | 6 个权重集成 | 折模型多样性 + 全量模型数据量 | 算力最多 |

推荐：**有 ≥4 张卡就走 A**（顺手拿到集成与 OOF）；算力紧或想快速迭代就走 B/C。
**A 与 B 可以叠加**：先 5 折拿到数字和集成权重，再用 `03_train.sh full` 补一个全量模型进集成。

### 4.3 用验证集也要有节制（乐观偏差）

- 在验证集上**只评估一次**→ 无偏；**反复**用它调超参 / 选模型 / 挑 TTA / 标阈值 → 指标会乐观（这就是"用验证集做题"）。
- 因此建议：超参用折内 val（OOF）定；官方验证集用来**确认**最终口径、标定阈值、做最终报告。
- 验证集通常只有几十~几百例，**单例波动大**：拿它当唯一早停依据容易抖（折内 val 一般更大更稳）。代码里两者都支持，默认以折内 val 早停。
- 无论哪种口径，`16_finalize.sh` 都保证**标定与评估同口径**（不会一个用 external、一个用折内），避免"阈值按 A 数据标、指标按 B 数据报"的错配。

### 4.4 全量训练怎么跑（B / C）

> 命令细节、期望输出与失败处理见 [`docs/TRACK4_RUNBOOK.md`](docs/TRACK4_RUNBOOK.md) §8（本表只做索引）。

| 想干什么 | 命令 | 照抄版 |
|---|---|---|
| 前置：接入官方验证集（全量训练唯一的无泄漏验证数据） | `export VAL_ROOT=/2026aicompetition/datasets/verification`；`bash scripts/01_probe.sh --val` → `data/manifest_val.json` | [手册 §8.3](docs/TRACK4_RUNBOOK.md) |
| 全量训练（B）：train=清单全部病例，val=官方验证集 | `bash scripts/03_train.sh full` → `checkpoints/g4_full/best.pth`（日志 `logs/train_full.log`） | [手册 §8](docs/TRACK4_RUNBOOK.md) |
| 多 seed（C） | `sed 's/^seed: 42/seed: 43/' configs/train.yaml > configs/train_s43.yaml`；`CONFIG=train_s43 TAG_PREFIX=g4_full43 bash scripts/03_train.sh full` | [手册 §8](docs/TRACK4_RUNBOOK.md) |
| 评估全量权重（**必须显式 `--ckpt`**） | `CKPT=checkpoints/g4_full/best.pth bash scripts/04_eval.sh --split external` | [手册 §10](docs/TRACK4_RUNBOOK.md) |
| 标定阈值 / 特殊影像 | `python scripts/14_calibrate_thresholds.py --split external --ckpt checkpoints/g4_full/best.pth --write`；`python scripts/15_eval_special_dup.py --split external --ckpt checkpoints/g4_full/best.pth` | [手册 §10](docs/TRACK4_RUNBOOK.md) |
| 全量模型集成（C/D） | `CKPT=checkpoints/g4_full/best.pth,checkpoints/g4_full43/best.pth bash scripts/04_eval.sh --split external` | [手册 §10](docs/TRACK4_RUNBOOK.md) |

两个必须记住的点：

- `bash scripts/03_train.sh full` 等价于 `python -m src.training.trainer --config train --fold full --tag g4_full`。
- **自动发现只认 `checkpoints/g4_fold*/`**：全量权重（`g4_full*`）不显式给 `--ckpt` 就会被忽略，评出来的是别的权重（数字对不上时先查这里）。

细节与护栏：

- 没有 `data/manifest_val.json` 时，`--fold full` **直接报错并打印上面两条接入命令**，不会静默拿训练集当验证集。
- 启动时会打印 `[trainer] 全量模式：train=N（清单全部病例） val=M（官方验证集 …）`。
- **选模只看 Dice，所以只有带掩膜的病例能进 val**（`external_val_cases` 只挑 `masks` 非空的例）：
  结构化字段标签在 val 里**不参与任何计算**，而没掩膜的例会让 `make_targets` 产出**全零 target**，
  Dice 在"预测也为空"时按 `den == 0` 记成 **1.0 的假满分**，会把 best 选歪。
  也因此：**验证集没有字段金标准不影响全量训练**，只要它有掩膜；没有掩膜时 `--fold full` 会直接报出来
  （不会再喊你去跑探针），改用折内 val（`bash scripts/03_train.sh 0`）。
- 全量模式的 `best.pth` 依赖验证集质量：验证集若只有几十例，建议**固定 epoch 数**而不是纯靠早停（`epochs` 在 `configs/train.yaml`）。
- 交付与评估口径仍是 external：`bash scripts/16_finalize.sh --print-split` 会显示当前判定，用于确认没有串口径。

---

## 5. 评估口径：external 与折内（留一/OOF）

### 5.1 口径怎么定（唯一入口）

判定只看一件事：**`data/manifest_val.json` 是否存在且能解析出病例**（即"官方验证集接进来了没"）。
三个评估脚本（`14` / `15` / `16`）共用同一判定，`--split auto`（默认）自动选：

| 判定 | 口径 | 评估数据 | 模型集合 |
|---|---|---|---|
| 有 `manifest_val.json` | **external** | 官方验证集 | **全部折**集成（不做留一） |
| 没有 | **折内/OOF** | 训练集内 val | 留一折集成（评折 `f` 时用其余折）/ OOF |

- 也可手工强制：`--split fold`（折内，必须给 `--fold`）、`--split external`、`--split oof`（目标一二/重复影像专用）、`--split all`（快速看趋势，会虚高）。
- 查看当前判定：`bash scripts/16_finalize.sh --print-split`（输出 `split=external` 或 `split=oof`）。
- 回退到旧口径：删掉 `data/manifest_val.json`，或清空 `VAL_ROOT` / `paths.yaml: raw.val`。

### 5.2 为什么两种口径的集成方式不同

| | 官方验证集（external） | 折内 val（无验证集时） |
|---|---|---|
| 与训练集关系 | 无交集 | 第 `f` 折的 val **被第 `f` 折模型见过** |
| 集成策略 | **全部折**（留一反而每条少用一个模型，白白丢分） | **留一**（评折 `f` 只用其余折模型） |
| 目标一二/重复影像 | 直接用验证集自带的 `special` 金标准 | **OOF**：逐折用本折模型只评本折 val 再汇总 |

**为什么目标一二/重复影像必须 OOF（没有验证集时）**：本任务正样本极稀疏（624 例中 fake 6 / Composition 6 / 重复对 12）。按单折 val 评，正样本只有 1~2 个、重复对甚至为 0 → AUC 失去统计意义；跑全量又会让折模型评到自己的训练集 → 虚高。OOF 让每一例都恰好被"没见过它的模型"预测一次，**无泄漏且用满全部样本**。

### 5.3 已知偏差（读数字前必看）

- 旧折划分（`folds.json` 由早期 612 例清单生成）存在 **3 例重复计数 / 2 例缺失**：评估脚本按 "fold 内 val" 聚合并打印数量，`eval_cases()` 在清单里查不到的病例会**报警并跳过**，不会静默改变分母。当前 `folds.json` **未重新生成**（重生成会让已有权重与划分不再对齐）。
- `04_eval.sh --limit N` 抽的是**前 N 例**，不是随机子集；比较不同配置时必须用同一个 `--limit`（或全量）。
- 官方验证集里**没有掩膜**的病例会被跳过分割、但**仍参与重复影像配对**（报告里的 `n_no_mask` 是跳过数）。
- 本地复现/演练数据（`data_source=local/*`）的权重**不能**当正式结果；`assert_data_source` 会在训练与评估入口拦截清单与本机数据根不同类的情况。

### 5.4 参考基线（本地复现数据，仅证明链路通畅）

留一折集成、本地复现集（单折 `n≈125`，**非官方数据**）：

| 折 | 集成权重 | dice_core | dice_peri | dice_mean | hd95_peri |
|---|---|---|---|---|---|
| 0 | 全部 | 0.888 | 0.941 | 0.914 | 6.50 |
| 1 | 全部 | 0.884 | 0.945 | 0.914 | 4.77 |
| 2（留一） | fold0 + fold1 | 0.875 | 0.933 | 0.904 | 6.40 |

重复影像（同一批本地数据）：学习式嵌入 OOF `auc_pr` 0.001~0.167（每折仅 0~2 个正对，**噪声极大，不可外推**）；
几何·强度指纹 `AUC=1.0000`（正对最小 0.9398 / 负对最大 0.6817，12 个正对）。**正式数字必须在官方验证集上重跑**。

---

## 6. 配置速查

改配置只改 `configs/*.yaml`；环境变量可临时覆盖路径类配置（见 2.4）。

### 6.1 `configs/paths.yaml`

| 键 | 默认 | 说明 |
|---|---|---|
| `workspace` | `/2026aicompetition/workspace` | 平台私有存储根（日志/答案/权重） |
| `raw.track4` | `/2026aicompetition/datasets/training` | 训练集根（可填 `.../annotation`，自动下钻） |
| `raw.val` | 空 | 验证集根；**填了就启用 external 口径** |
| `manifest` / `manifest_val` | `data/manifest.json` / `data/manifest_val.json` | 训练/验证清单 |
| `folds` | `data/folds.json` | 折划分 |
| `preprocess_cache` | `data/preprocess_cache` | 1mm 公共网格 + 整脑视图缓存 |
| `logs_dir` / `checkpoints_dir` | `logs` / `checkpoints` | 平台环境下日志自动切 `{workspace}/logs` |
| `answer_root` | `/2026aicompetition/workspace/answer` | 推理结果（`{evaluation_id}/`） |

> 验证集实测布局是 `verification/original/`（影像与 `SeriesType.xlsx` 同层）：
> `VAL_ROOT` 填 `…/verification` 或 `…/verification/original` 都能扫通。
>
> **验证集只有 `SeriesType.xlsx`，没有字段金标准表** —— `脑胶质瘤标注结果-训练集.xlsx` 与
> `脑胶质瘤标注结果-验证集.xlsx` 都不在。所以验证集病例的 `labels` 全为空、探针报告里
> `no_labels` 会列出全部病例（提示语也改成了"这是预期"而不是"去找表"）：分类头只由**训练折**
> 监督，官方评估口径是 Dice/NSD/HD95 + 重复影像（`scripts/04_eval.sh --split external`），
> 没有 `duplicate_w*` 键即表示验证集没给重复影像金标准。平台若后续单独发布验证集标注表：
> 放进验证集目录任意位置（或 `export GLIOMA_LABELS_DIR=<含表目录>`），重跑 `01_probe.sh --val` 即自动接上。
> 另：**"全量训练"（`03_train.sh full`）的早停集需要验证集病例带掩膜**（字段金标准对早停无意义）——
> 验证集若连掩膜也没有，`full` 模式会直接报出来（而不是让你反复重跑探针），改用折内 val 即可。

### 6.2 `configs/train.yaml`（常用项）

| 键 | 默认 | 说明 |
|---|---|---|
| `seed` / `epochs` | 42 / 100 | 多 seed 集成时复制本文件改 `seed` |
| `batch_size` / `patch_size` | 2 / `[96,96,96]` | 显存不足 → `80³` 或 batch 1 |
| `lr` / `weight_decay` | 3e-4 / 3e-5 | |
| `amp_dtype` | `bfloat16` | 平台 CUDA12.8 支持；老卡改 `float16` |
| `ema_decay` / `grad_clip` | 0.999 / 12.0 | 推理默认用 EMA 权重（`infer.use_ema`） |
| `folds` / `val_ratio` | 5 / 0.2 | 折数与折内验证比例 |
| `val_every` / `checkpoint_metric` | 1 / `val_dice_peri` | 选模指标（新增官方验证集时可改，注意同口径） |
| `threshold_grid` | 0.10~0.90（18 档） | 阈值标定搜索范围（`14_calibrate_thresholds.py`） |
| `model.*` | `mednext` / base32 / depth4 / k3 | `arch: resunet` 可回退轻量版 |
| `sampler.pos_ratio` | 0.7 | 含病灶 patch 比例（病灶极小时可调高） |
| `augment.*` | 见文件 | 小样本泛化主要来源（BraTS 类通常 +2~4 Dice） |
| `global_view.size_mm/out/every` | 192 / 96 / 4 | 全局头物理立方体边长；**训练与推理必须一致** |
| `aux.special_batch/pair_batch/neg_per_pos` | 2 / 2 / 3 | 目标一二与重复影像的每步样本量 |
| `loss.*` | dice 1.0 / ce 1.0 / cls 1.0 / special 1.0 / embed 0.3 / contain 0.2 | 含 `deep_sup`、`boundary`、各类 `pos_weight` |
| `infer.use_ema/ensemble` | true / true | 推理用 EMA + 多折集成 |

另有 `configs/train_large.yaml`（大模型/长训练）与 `configs/_smoke_train.yaml`（冒烟用）。

### 6.3 `configs/preprocess.yaml`

| 键 | 默认 | 说明 |
|---|---|---|
| `geometry.common_spacing` | `[1,1,1]` | 公共网格：各序列先对齐到 1mm 各向同性 |
| `geometry.crop_brain` / `brain_margin_vox` | true / 4 | 推理按脑部包围盒裁 ROI（提速） |
| `geometry.resample_order_mask` | 0 | 掩码**必须**最近邻（保持 0/1） |
| `intensity.method` | `zscore_per_volume` | MRI 标准做法（非 CT 固定窗）；`foreground_only` 只用非零体素估计 |
| `channels` | t1c/flair/t2/t1 | 缺序列时按 `fallback` 取替代（t1c←t1/t2、flair←t2） |
| `mask_roles.core/peri/abn` | 见文件 | 角色按**掩码所在序列的模态**归位（任务A=T1C 核心区、任务B=FLAIR/T2 异常区） |
| `inference.patch` | `[96,96,96]` | **必须与训练一致**，否则明显掉点 |
| `inference.overlap` / `seg_tta_flips` / `tta_batch` | 0.4 / x,y / 2 | 滑窗重叠与 TTA（耗时主体） |
| `inference.global_size` | 96 | **必须与 `train.global_view.out` 一致** |
| `inference.min_tumor_voxels` / `keep_components` / `bridge_mm` | 30 / 3 / 10.0 | 后处理：去小碎片、保留多灶、邻近连通域合并 |
| `inference.duplicate_topk` / `duplicate_per_study` / `fp_weight` | 200 / 50 / 0.5 | 重复影像：每例提交对上限、实际提交数、手工指纹权重 |
| `inference.prefetch_workers` | 2 | I/O 预取线程（DICOM→NIfTI/重采样与 GPU 推理重叠） |
| `special.*` | 0.05 / 0.35 | 特殊影像启发式兜底阈值（脑组织占比、边缘不连续比例） |

### 6.4 日志字段（规范要求，别改格式）

`{workspace}/logs/*.jsonl`，每行含
`timestamp / epoch / step / phase / mode / loss / lr / data_source / checkpoint / pretrained_from`。

---

## 7. 常见问题

### 7.1 报 `ValueError`：数据根指向了多阶段父目录

把 `DATASET_ROOT` 改成具体阶段目录（`/2026aicompetition/datasets/training`）。
平台实测：`training/` 下**只有** `annotation/`，影像与 `SeriesType.xlsx` 都在 `training/annotation/`（官方 **3255 例**；本地复现集 612 例）。
填 `training` 会**自动下钻并告警**，填 `datasets` 会被拦住。

### 7.2 病例数正常，但报"无任何可用序列"（模态全是 `other`）

说明 `SeriesType.xlsx` 没被找到。

```bash
python scripts/30_inspect_table.py --root $DATASET_ROOT --rows 3   # 确认表在不在、列名对不对
export GLIOMA_LABELS_DIR=<表所在目录>                               # 显式指定（团队工作区里的表这类情况）
bash scripts/01_probe.sh                                            # 看 series_type_rows / modality_counts
```

注意只认数据集自带的 `SeriesType.xlsx`；`3_serieslabel.xlsx` 与本赛道无关（会把 `T2WI`/`T2-Flair` 覆盖成 `T2`）。

### 7.3 显存不足

按顺序试：`batch_size` 2→1 → `patch_size` `96³`→`80³` → 关 TTA（`seg_tta_flips: []`、`tta_batch: 1`）→
`inference.overlap` 0.4→0.3 → `num_workers` 降到 2（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已在脚本里设置）。
用 `python scripts/12_bench.py` 先量出真实峰值再定。

### 7.4 断点续训 / 重训

- 训练中断后直接重跑同一命令：会自动读 `checkpoints/<tag>/last.pth` 续训（`--no-resume` 可关闭）。
- 换数据必须重建缓存（`13_build_cache.py --force`）与清单（`01_probe.sh`），否则读到旧网格。
- 从头再来：删除 `checkpoints/<tag>/` 与对应 `logs/<tag>.jsonl`。

### 7.5 无网环境

```bash
# 联网机
bash scripts/00b_prepare_wheels.sh --with-torch --mirror tsinghua --out wheels
# 拷贝 wheels/ 到无网机
bash scripts/00_setup_env.sh --mode system --offline --wheels wheels
```

### 7.6 推理服务相关

- 单例全流程（含 TTA + 集成 + 后处理）**实测约 165 秒**：所以 `/call` 必须 5s 内先回 200、后台异步跑；
  回调前的等待要留足（否则会在推理结束前超时）。`06_platform_serve.sh` 已按此实现，`MAX_CONCURRENT_JOBS` 默认 2。
- 回调地址**必须**用 `CALLBACK_URL` 传入（容器页面上方有完整地址）；脚本也会尝试若干常见变量名。
- 加载容错：`GLIOMA_LOADER_TOLERANT=1`（`06` 默认开）——坏文件跳过而不是整批失败。
- 本机自测：`bash scripts/05_serve.sh`（`PORT` 默认 8000，含 `/health` 与一次 `/call` 冒烟）。

### 7.7 提交前检查的预期结果

- 本地（无官方数据、未训模型）跑 `bash scripts/23_pre_submit_check.sh --quick` 会看到**少数失败项**，
  固定是"权重未导出"这类容器内步骤——**必须在容器内做完导出后转绿**，它正是"忘导权重就提交"的最后防线。
- 权重交付路径：`{workspace}/checkpoint/{goal}/model.pt`，由提交工程在**服务启动时**读取。

### 7.8 其它

- `data/` 里若带本地复现清单（`data_source=local/*`），正式训练前用 `bash scripts/17_reset_for_official.sh` 归档，再按 2.6 切到官方数据根。
- `logs/`、`checkpoints/` 里的历史产物只作参考；**正式数字必须来自 official 数据根 + 当前代码**。
- 演练（`_smoke_train` / `99_smoke_test`）产物权重**不能**提交。

---

## 8. 附录

### 8.1 脚本清单

**主流程**

| 脚本 | 作用 | 主要参数 / 环境变量 |
|---|---|---|
| `00_setup_env.sh` | 环境安装/体检 | `--mode {auto,conda,system}`、`--offline`、`--wheels DIR`、`--mirror`、`--name`、`--python`、`--with-torch`、`--torch-index` |
| `00b_prepare_wheels.sh` | 联网机下载 wheel（供离线安装） | `--out DIR`、`--with-torch`、`--mirror`、`--python` |
| `01_probe.sh` | 数据探针 → 清单 | `[数据根]`、`--val`（验证集 → `manifest_val.json`） |
| `02_build_dataset.sh` | 分层多折划分 | `N_FOLDS`（默认 5，**不要**用 `FOLDS`） |
| `03_train.sh` | 训练 | `0`/`all N`/`one`/`full`、`FOLD`、`TAG_PREFIX`、`CONFIG`、`PRETRAINED`、`PY` |
| `13_build_cache.py` | 预处理缓存 | `--root`、`--cache`、`--workers`、`--force` |
| `12_bench.py` | 速度/显存基准 | `--root`、`--patch`、`--batch`、`--steps`、`--eval-n`、`--infer-patch`、`--folds` |
| `14_calibrate_thresholds.py` | 集成阈值标定（写回各折） | `--fold`、`--split`、`--ckpt`、`--limit`、`--grid`、`--write` |
| `04_eval.sh` | 全图分割评估 | `[fold] [limit]`、`--split`、`CKPT` 环境变量 |
| `15_eval_special_dup.py` | 目标一二 + 重复影像评估 | `--fold`、`--split`、`--ckpt`、`--limit`、`--fp-weight` |
| `16_finalize.sh` | 收尾：标定 → 评估 → 导出 | `FOLDS`、`SKIP_MOCK=1`、`--print-split` |
| `09_export_submission.sh` | 导出提交权重（规范 §5.2） | `[folds]`、`--verify`、`GLIOMA_CHECKPOINT_ROOT` |
| `08_mock_competition.sh` | Mock Competition（协议闭环 + 指标复算） | `[folds]`、`GLIOMA_CKPT` |
| `05_serve.sh` | 本机起服务并自测 | `PORT`（默认 8000）、`GLIOMA_CKPT` |
| `06_platform_serve.sh` | 平台容器起服务（回调） | `WORKSPACE`、`PORT`、`HOST`、`CALLBACK_URL`、`MAX_CONCURRENT_JOBS`、`GLIOMA_LOADER_TOLERANT` |
| `07_export_weights.sh` | 导出权重（服务/交付） | 见脚本头 |

**数据准备与运维**

| 脚本 | 作用 |
|---|---|
| `29_locate_dataset_root.py` | 定位真正的数据根（`--base`、`--goals`、`--max-depth`） |
| `30_inspect_table.py` | 摊开看数据信息表（`[表文件]`/`--root`、`--rows`、`--cols`） |
| `31_train_modality_model.py` | 训练"序列类型"兜底判别模型（模态表缺失时） |
| `32_apply_abnormal_patch.py` | 标签表修补核验/合并（`--patch`、`--apply`、`--dry-run` 语义见 `--help`） |
| `33_fix_abnormal_labels.sh` | 一键修 `1_abnormal` 标签表并重跑探针（`--dry-run`） |
| `10_brats_to_track4.py` | 公共 BraTS → track4 目录结构（本地实验） |
| `17_reset_for_official.sh` | 归档本地产物，切官方数据前清场 |
| `18_retrain_new_folds.sh` / `19_safe_retrain_pipeline.sh` | 重训新折 / 安全重训流水线 |
| `11_dicom_selftest.py` | DICOM 读取链路自检 |

**自检与校验**（提交前建议全跑）

| 脚本 | 作用 |
|---|---|
| `99_smoke_test.py` | 合成数据端到端冒烟（答案格式合规） |
| `20_bridge_selftest.py` | 与提交工程的坐标/几何桥接自检 |
| `21_verify_fixes.py` | 回归修复项自检 |
| `22_fault_injection.py` | 故障注入（坏文件/缺模态/空预测） |
| `23_pre_submit_check.sh` | 提交前总检查（`--quick` 跳耗时项） |
| `24_verify_eval_split.py` | 评估划分语义自检（external / 折内 / OOF） |
| `25_verify_tasks_integration.py` | 提交侧插件集成校验（需兄弟仓库在旁） |
| `26_audit_plugin_completeness.py` | 插件完整性审计 |

### 8.2 自检清单（交付前逐条打勾）

- [ ] `python scripts/99_smoke_test.py` 通过
- [ ] `bash scripts/01_probe.sh` 的 `case_count` / `series_type_rows` / `modality_counts` 正常
- [ ] `python scripts/24_verify_eval_split.py` 全部 PASS（口径判定、无掩码跳过、每例都被评测到）
- [ ] `python scripts/21_verify_fixes.py`、`python scripts/22_fault_injection.py` 通过
- [ ] `bash scripts/23_pre_submit_check.sh` 全绿（本地允许"权重未导出"一项，容器内做完导出后必须转绿）
- [ ] `bash scripts/09_export_submission.sh --verify` 通过（六目录 + 规范文件名）
- [ ] `{workspace}/logs/*.jsonl` 字段齐全、路径在 `{workspace}/logs`
- [ ] 掩码体素严格 0/1、仿射/维度与对应模态一致（`write_mask` 已复核，抽查 1~2 例）

### 8.3 红线

- 掩码写盘必须与**对应模态原图**同仿射同尺寸；体素只能是 0/1 —— 违反该例分割直接 0 分。
- 日志必须写 `{workspace}/logs`，字段名按规范（评测会校验）。
- `/call` 必须 5s 内回 200；结果写 `answer/{evaluation_id}/`，`prediction.json` / `duplicate_pairs.jsonl` 格式与命名按规范。
- 不得使用与赛道四无关的标注表（`3_serieslabel.xlsx` 等）推断模态。
- 本地演练数据训出的权重不得当作正式交付结果。
