# 赛道4 · 基于 MRI 的脑胶质瘤智能检测（自建模型组）

> 严格对齐《赛事开发规范（赛道四）V1.0》《公共数据集格式说明》
> 《云电脑平台使用指南V1.0》《训推平台使用指南V1.0》《赛事管理平台使用指南V2.0》。
>
> 本工程的**审核结论、缺陷清单与升级说明**见 [`docs/REVIEW_AND_UPGRADE.md`](docs/REVIEW_AND_UPGRADE.md)；
> 逐条合规自检见 [`docs/SPEC_CHECKLIST.md`](docs/SPEC_CHECKLIST.md)。

---

## 0. 任务与评分（规范原文 → 本工程实现）

| 目标 | 规范要求 | 本工程实现 |
|---|---|---|
| **目标一** 假人体 | `prediction.json.IsNotHumanBodyProb` | `special` 头（`annotation/fake` 监督）+ 保守启发式兜底 |
| **目标二** 拼接影像 | `prediction.json.IsStitchedProb` | `special` 头（`annotation/Composition` 监督）+ 启发式兜底 |
| **重复影像** | `duplicate_pairs.jsonl`：`{StudyUID, StudyUID_dup, PairProb}`，**每例 ≤200 对**、对序自由、缺省 0；指标 Recall@10%FPR / Precision@15%Recall / AUC-PR | 学习式嵌入（金标准正/负对训练）**＋** 几何·强度指纹融合 → 标定 → 每例 Top-200 |
| **任务A** | **T1 增强核心区**二值掩码；仿射与维度**必须与 T1C 原图一致**；体素严格 0/1 | `predict` 滑窗 → 阈值 → 连通域后处理 → 回采样到 T1C 原始空间 → 写盘后回读复核 |
| **任务B** | **FLAIR/T2 总异常区**二值掩码；仿射与维度**必须与 FLAIR/T2 原图一致** | 同上（回采样到 FLAIR/T2 原始空间），并强制 `peri ⊇ core` |
| **结构化诊断** | `Prediction` 14 字段（`TumorProbability / Location / Morphology / WHO_Grade / Enhancement / EnhancementPattern(8 类) / Necrosis / CysticChange / Hemorrhage / Calcification / Margin / Lobulation / Signal_T2WI / Signal_FLAIR`）+ `Interpretation.Conclusion` | 多任务头（逐字段 `label_mask` 稀疏监督）+ 规则校正 |
| 日志 | JSONL，路径**必须** `/2026aicompetition/workspace/logs`，字段 `timestamp(ISO8601…Z)/epoch(从1)/step/phase/mode/loss/lr/data_source/checkpoint/pretrained_from` | `src/utils/logger.py`（平台目录自动切换） |
| 推理服务 | `POST /call`（嵌套 `input:{evaluation_id,dataset_path}`）、**5s 内立即 200**、`GET /health` 200、结果写 `answer/{evaluation_id}/`、**回调** `{平台}/api/competition/inference/callback/` 带 `request_id + evaluationId + predPath` | `src/serving/app.py`（后台异步 + 启动预检 + 回调重试 + `/status` 自测） |
| 强制提交格式 | 掩码体素**严格 0/1**、仿射/维度与对应模态一致，否则该例分割**直接 0 分** | `write_mask` 写盘后**回读复核** + `validate_answer` 复检 |

---

## 1. 上分点（相对初版的关键增强）

1. **目标一/二真正被训练**（初版 `special` 头从未被任何损失监督——基础考核项不达标直接淘汰）；
2. **重复影像嵌入真正被训练**（初版嵌入损失恒被跳过），并叠加几何/强度指纹；
3. **修复 TTA 双重翻折**（初版会把方向错误的预测平均进结果）；
4. **1mm 各向同性公共网格**真正生效（初版配置项从未被使用）；
5. **掩码按序列模态归位**（初版会把 FLAIR 上的"瘤体"误判为任务A的 core）；
6. **掩码 header 去污染**（初版继承 `scl_slope` 会让掩码回读值 ≠ 0/1 → 该例 0 分）；
7. **DICOM 支持**（规范原文写"影像原始数据为dicom格式"）；
8. **兜底答案**：单例异常也写出合规文件，避免"整例缺文件"归零；
9. MedNeXt 风格骨干 + 7 类强增广 + 逐通道阈值 + 连通域后处理 + 预取流水线。

---

## 2. 目录结构

```
glioma_track4/
├── configs/                 paths / preprocess / train / labels
├── src/
│   ├── utils/               配置加载 + 规范 JSONL 日志（平台路径自动切换）
│   ├── data/                probe（探针）、dicom（DICOM→NIfTI）、dataset（网格/增广/数据集）、labels
│   ├── models/              mednext（MedNeXt 风格骨干）+ unet3d（ResUNet 备选与多任务头）
│   ├── training/            损失 + 指标 + 训练器（AMP/EMA/断点/多折/四路损失）
│   ├── inference/           滑窗+TTA+集成、病例流水线、结构化推导、重复匹配、答案写出
│   └── serving/             FastAPI 服务（/call + /health + 回调）
└── scripts/
    ├── 00_setup_env.sh      依赖安装（离线/平台容器）
    ├── 01_probe.sh          探针：实测数据结构 → manifest
    ├── 02_build_dataset.sh  分层折划分
    ├── 03_train.sh          多折训练（支持 PRETRAINED）
    ├── 04_eval.sh           折内 Dice/NSD/HD95 + 重复影像三项指标 + 阈值扫描
    ├── 05_serve.sh          本地服务与 /call 契约自测
    ├── 06_platform_serve.sh **测评容器启动命令**（校验回调地址）
    ├── 07_export_weights.sh 权重/日志导出到平台持久化目录
    ├── 10_brats_to_track4.py 公开数据 → 赛道四格式（**仅本地实验**）
    ├── 11_dicom_selftest.py DICOM 路径端到端自检
    ├── 12_bench.py          训练/推理速度与显存基准
    ├── 13_build_cache.py    预处理缓存（训练吞吐关键）
    └── 99_smoke_test.py     合成数据端到端自检（含答案格式强制校验）
```

---

## 3. 快速开始

```bash
# ① 环境
bash scripts/00_setup_env.sh                      # 联网机器
bash scripts/00_setup_env.sh --mode system --offline --wheels wheels   # 云桌面（无网）

# ② 自检（无需真实数据，验证全链路与答案格式）
python scripts/99_smoke_test.py

# ③ 数据
bash scripts/01_probe.sh                          # 探针 → data/manifest.json
bash scripts/02_build_dataset.sh                  # 分层折划分 → data/folds.json
python scripts/13_build_cache.py --workers 8      # 预处理缓存（强烈建议，训练提速数倍）

# ④ 训练与评估
bash scripts/03_train.sh all 4                    # 4 折并行（4 张卡）
bash scripts/04_eval.sh 0 30                      # 折内指标

# ⑤ 本地服务自测
CKPT=checkpoints/g4_fold0/best.pth bash scripts/05_serve.sh
```

平台（测评容器）**启动命令**填：

```bash
CALLBACK_URL="<容器实例页面上方显示的完整回调地址>" \
  bash /2026aicompetition/workspace/glioma_track4/scripts/06_platform_serve.sh
```

---

## 3.5 与团队串联提交工程衔接（协作模式）

本工程按《赛道四_自建模型组_系统架构与协作规范》作为**算法工程**接入团队的
`Glioma_recognition` 提交入口——协议、Writer、Validator、回调由团队仓库单点维护，
本工程只提供 `StudyTask` / `DatasetTask` 插件与 checkpoint。

```bash
# ① 导出权重到规范约定的目录（§5.2）
python -m integration.export_ckpt --folds g4_fold0,g4_fold1,g4_fold2 --mode link
python -m integration.export_ckpt --verify

# ② 桥接自测（完全走团队的 Loader/Runner/Writer/Validator）
python scripts/20_bridge_selftest.py --n 3

# ③ 提交入口侧打开插件
export COMPETITION_PIPELINE_FACTORY=tasks.glioma.pipeline:build_pipeline
```

详见 [`docs/INTEGRATION_WITH_TEAM.md`](docs/INTEGRATION_WITH_TEAM.md)（含 6 个已处理的
契约陷阱与 5 项待确认风险）。

---

## 4. 平台/合规要点

| 事项 | 做法 |
|---|---|
| 云桌面**无互联网** | `bash scripts/00_setup_env.sh --mode system --offline --wheels wheels` |
| 训练日志 | 自动写 `/2026aicompetition/workspace/logs/*.jsonl`；`data_source` 按**实际数据根**自动生成（`official/*` 或 `local/*`），不硬编码 |
| 结果目录 | `/2026aicompetition/workspace/answer/{evaluation_id}/`（`<AccessionNumber>/prediction.json` + `<AccessionNumber>/<SeriesUid>/*.nii.gz` + `duplicate_pairs.jsonl`） |
| 权重持久化 | `bash scripts/07_export_weights.sh` → `/2026aicompetition/workspace/train_model/` |
| 容器数量 | 4 个上限 → 4 折并行 + 第 5 折后补 |
| **回调闭环** | 通过 `CALLBACK_URL`（或 `{workspace}/callback_url.txt`）传入平台回调地址。缺失时 `/health` 返回 `callback_ready:false`、启动打印醒目告警、回调处**显式报错**——不会静默不回调 |
| **数据源卫生** | 正式权重只能来自大赛数据：切到官方数据前执行 `bash scripts/17_reset_for_official.sh` 归档本地验证产物并重建清单；`assert_data_source` 闸门会拒绝"清单数据源与本机不同类"的训练 |

### 4.0 平台数据目录与数据根选择

容器内 `/2026aicompetition/datasets` 下是**五个平行阶段目录**：

```text
/2026aicompetition/datasets/
├── training/            ← 官方训练集（数据根取这一层）
├── evaluation_first/    ← 第一轮评测输入（推理用）
├── evaluation_second/   ← 第二轮评测输入（推理用）
├── evaluation_finals/   ← 决赛评测输入（推理用）
└── verification/        ← 验证集
```

**数据根必须精确到阶段目录**：`DATASET_ROOT=/2026aicompetition/datasets/training`。
停在上层是很容易犯的错，而且后果不会当场显现——阶段名会被当成检查号，
清单里出现 5 个假病例、金标准一张也对不上，训练却照常跑完。
现在两条训练路径都会**在扫描前直接失败**并给出应填的路径：

```text
ValueError: 数据根 /2026aicompetition/datasets 指向数据集父目录，
其下是平台阶段目录 ['evaluation_finals', 'evaluation_first', ...]。
请把数据根设为具体阶段（训练应为 /2026aicompetition/datasets/training）；…
```

护栏位置：`shared/data.py: assert_case_root()`（研发侧各 Goal 的 `discover_cases`）
与 `src/data/probe.py: assert_case_root()`（算法工程的探针/缓存/推理入口）。

> 顺带一提：`training/` 与 `verification/` 内都可能有 `annotation/`
> （`fake` / `Composition` / `duplicate`），它**不是**检查号目录；
> 扫描时会被跳过，只有 `annotation/{fake,Composition}` 中的病例会作为
> 特殊影像正样本补进清单。

**模态与掩膜来自 `SeriesType.xlsx`。** 官方数据的序列目录名/文件名是 DICOM UID，
靠关键词猜不出模态；数据根（与 `<检查号>/` 同级）必须有 `SeriesType.xlsx`，
探针报告里的 `series_type_rows` 应大于 0、`modality_counts` 应出现 t1c/flair/t2/t1。
否则表现为"病例数正常、却报 `无任何可用序列`"或整批归到 `other`。

> 遇到数据根/模态相关的问题，先按 `docs/DATASET_ROOT_TROUBLESHOOT.md` 排查：
> 定位数据根 → `python scripts/29_locate_dataset_root.py`。

### 4.1 从本地验证切到官方数据（必须执行的顺序）

```bash
bash scripts/17_reset_for_official.sh                    # ① 归档本地验证产物（不删除）
export DATASET_ROOT=/2026aicompetition/datasets/training # ② 官方训练集根（含 annotation/ 与各检查号目录）
export CACHE_DIR=/2026aicompetition/workspace/cache      #    缓存放私有存储（容器删除不丢）
bash scripts/01_probe.sh                                 # ③ 重新探针（写入 data_source=official/train_v1）
bash scripts/02_build_dataset.sh                         # ④ 分层折划分
python scripts/13_build_cache.py --workers 8             # ⑤ 预处理缓存
bash scripts/03_train.sh all 4                           # ⑥ 4 折并行训练
FOLDS="0 1 2 3 4" bash scripts/16_finalize.sh            # ⑦ 阈值标定+评估+导出权重
```

### 4.2 训练完成后的收尾与提交（推荐顺序）

```bash
# ① 一键收尾：集成阈值标定 → 留一折集成全图评估 → 目标一二/重复评估
#            → 规范 §5.2 导出权重 → Mock Competition → 提交前检查清单
FOLDS="0 1 2" bash scripts/16_finalize.sh

# 也可分步执行：
bash scripts/14_calibrate_thresholds.py --fold 0 \
     --ckpt checkpoints/g4_fold0/best.pth,checkpoints/g4_fold1/best.pth,checkpoints/g4_fold2/best.pth \
     --limit 0 --write                        # 集成阈值标定（写回所有折，保证集成生效阈值一致）
python -m src.evaluation.evaluate --fold 0 \
     --ckpt checkpoints/g4_fold1/best.pth,checkpoints/g4_fold2/best.pth   # 留一折集成（无泄漏）
bash scripts/09_export_submission.sh 0,1,2   # 规范 §5.2 导出提交权重
bash scripts/08_mock_competition.sh          # 平台协议闭环（/health→/call→后台→回调→校验）
bash scripts/23_pre_submit_check.sh          # 提交前检查清单（9 项，全绿才提交）
```

**三个关键设计说明**

1. **留一折集成评估**：评估第 `f` 折时**只用其余折的模型**。若直接用全部折，
   第 `f` 折的模型见过本折验证集 → 数字会虚高。
2. **集成阈值必须重新标定**：`load_ensemble` 对各折阈值**取均值**，
   只有把标定结果写回**所有折**，集成实际生效的阈值才等于最优值。
3. **目标一/二与重复影像用 OOF 评估**（`15_eval_special_dup.py` 默认）：
   本任务正样本极稀疏（624 例中 fake 6 / Composition 6 / 重复对 12）。
   按单折 val 评，正样本只有 1~2 个、重复对甚至为 0 → AUC 失去统计意义；
   跑全量又会让折模型评到自己的训练集 → 虚高。
   **OOF（out-of-fold）**逐折用本折模型只评本折 val 再汇总：每一例都恰好被
   "没见过它的模型"预测一次，**无泄漏且用满全部样本**。
   若需快速看趋势可用 `--all`（会虚高，仅作参考）。

   > 实测（3 折 OOF，375 例）：fake AUC 0.9935、stitch AUC 0.9933；
   > 重复影像 `w_fp=0.5`（嵌入+几何强度指纹融合）AUC-PR **0.600**、
   > Precision@15%Recall **1.000**，而纯嵌入（`w_fp=0`）仅 0.120 —— 指纹融合是重复影像的关键。

**评估划分的已知偏差（当前保留，影响已量化）**

旧版 `build_folds` 用"轮转取模"拼 val 集，在 `round(len(pos)×0.2)×5 ≠ len(pos)` 时
必然出错。实测 624 例 / 5 折下：**3 例重复**出现在两个折的 val（`FAKE_000/001`、`COMP_002`），
**2 例从未进入任何 val**（`BraTS2021_00085/00318`）。

- 根因已修（现为标准的"轮流发牌"分区，`scripts/24_verify_eval_split.py` 可验证）；
- 但**当前不重新生成 `folds.json`**：已训练的 4 个折基于旧划分，改划分会让旧权重
  评到自己训练过的病例（泄漏）。因此保留原划分，在评估侧**去重**并显式提示。
- 影响量级：OOF 覆盖 622/624 ≈ 99.7%，对指标的影响可忽略。

**若要启用新划分（需完整重训）**

```bash
bash scripts/18_retrain_new_folds.sh     # 停训练→归档→探针→新划分(并校验)→缓存→4折并行
```

- **必须从头重训**：旧权重的训练集基于旧划分，新旧 val 重叠约 80%，
  拿旧权重按新划分评估会严重泄漏 —— 两者不可混搭。
- 脚本第 4 步会**当场校验**新划分无重复、无缺失，不干净直接退出。
- `build_folds` 现在**默认复用**已有划分（`trainer` 每次启动都会调用它；
  若无条件重写，重启任意一折就会静默换掉整套划分，且不报任何错）。
  需要重建时必须显式 `force=True` 或先删除 `folds.json`。

**无人值守·安全重训（先保险，再重训）**

```bash
export DATASET_ROOT=/path/to/train_root
export BACKUP_ROOT=$HOME/glioma_submission_backup
nohup bash scripts/19_safe_retrain_pipeline.sh > logs/19_pipeline.log 2>&1 &
tail -f logs/19_pipeline.log
```

流水线 8 步：等 fold1 完成 → **导出 3 折保险** → 校验保险 → 停对照实验 → 归档旧产物 →
重新探针+新划分（**校验不干净即终止**）→ 重建缓存 → 启动 4 折重训。

保险的意义：新划分重训是 14~19h 的长任务，且归档不可逆。先把现有折按规范导出到
独立目录，无论重训成败，手上始终有**完整可提交**的版本。回滚方式：

```bash
export GLIOMA_CHECKPOINT_ROOT=$BACKUP_ROOT/checkpoint    # 指回保险即可
```

**环境差异（本地 vs 平台）**

| 变量 | 本地演练 | 平台容器 |
|---|---|---|
| `DATASET_ROOT` | 指向本地数据（如 `/mnt/.../track4_sim`），否则被"数据源合规闸门"拦截 | 默认 `/2026aicompetition/datasets/training` |
| `WORKSPACE` | 建议用 `/tmp/ws_test` 之类的临时目录 | 默认 `/2026aicompetition/workspace` |
| `GLIOMA_LOADER_TOLERANT` | 可选（默认关闭=严格模式，便于及早发现脏数据） | **`06_platform_serve.sh` 默认开启** |

**多折集成的部署优势**：无需手动指定权重——`resolve_checkpoints()` 会自动读取
`checkpoint/goal5_segmentation/` 下的**全部** `.pt` 做集成。

---

## 5. 严禁触碰的"红线"（规范明写）

1. 掩码**不得**出现 0/1 以外的值；仿射/维度**不得**与对应模态不一致 → 该例分割 **0 分**；
2. `duplicate_pairs.jsonl` 中 UID **必须**来自测试集检查列表；同一对重复出现取 `PairProb` 最大；文件至少一行；
3. `/call` **必须 5s 内**回 200（本工程为后台异步）；`/health` 必须常活（启动探针 5s/次、最长 1800s）；
4. 训练日志**必须**位于规定路径且为 JSONL；
5. 每个测试 `AccessionNumber` **必须**有 `prediction.json`（本工程对异常病例写兜底答案）。
