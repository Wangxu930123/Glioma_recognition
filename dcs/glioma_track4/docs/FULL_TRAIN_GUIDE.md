# 全量训练全流程说明（数据准备 → 训练 → 验证 → 交付）

> 适用：**不做交叉验证**、用"训练集全部病例 + 官方验证集"训一个模型（`--fold full`）。
> 本文只讲这一条路；折内 CV 的口径见 [`TRACK4_RUNBOOK.md`](TRACK4_RUNBOOK.md) 与 `README.md` §4/§5。
>
> 文中所有命令都在**项目根目录**执行（`cd <工程根>`）。

---

## 0. 三十秒速览

```bash
# ① 数据准备（训练集 + 验证集，各跑一次探针）
export DATASET_ROOT=/2026aicompetition/datasets/training
bash scripts/01_probe.sh                       # → data/manifest.json
export VAL_ROOT=/2026aicompetition/datasets/verification
bash scripts/01_probe.sh --val                 # → data/manifest_val.json【全量训练的硬前置】

# ② 体检：确认这两个数字（下面是判据，别只看"跑通了"）
#    manifest.json     : modality_counts 有 t1c/flair/t2 …
#    manifest_val.json : mask_role_counts 有 core/peri …   ← 没有掩膜就没法选 best

# ③（可选，提速）预构建 1mm 公共网格缓存
python scripts/13_build_cache.py --root "$DATASET_ROOT" --workers 8

# ④ 全量训练
bash scripts/03_train.sh full                  # → checkpoints/g4_full/{best,last}.pth

# ⑤ 一键收尾：标定阈值 → 验证集评估 → 目标一二 → 导出权重 → 提交前检查
FOLDS=full bash scripts/16_finalize.sh
```

**三条铁律**

1. **没有验证集就不要走全量**（`03_train.sh full` 会直接报错，不会静默拿训练集当验证集）。
2. **验证集必须有掩膜**（`*_mask.nii.gz`）；字段标注表（`脑胶质瘤标注结果-*.xlsx`）**不需要**。
3. 全量权重是 `checkpoints/g4_full/`，**自动发现只认 `g4_fold*`** —— 评估/导出时要显式给 `--ckpt` 或 `FOLDS=full`。

---

## 1. 先对齐概念

### 1.1 全量 vs 折内交叉验证

| | 折内 CV（`03_train.sh 0` / `all`） | **全量（`03_train.sh full`）** |
|---|---|---|
| train | 清单里 `1 - val_ratio` 的病例 | **清单全部病例** |
| val | 同一份清单的留出折 | **官方验证集**（与训练集无交集） |
| 早停 / 选 best | 折内 val Dice | 验证集 Dice |
| 权重目录 | `checkpoints/g4_fold0/` … | `checkpoints/g4_full/` |
| 折划分 `folds.json` | 需要（`02_build_dataset.sh`） | **不需要**（全量不读它） |
| 用途 | 出 OOF、量"没有验证集金标准的指标" | **最终交付**（数据一点不浪费） |
| 代价 | 每折少用 20% 数据 | 没有 OOF（选不出"没见过这例"的模型） |

两者**不冲突**：常见做法是全量出交付权重，折内模型用来量指标 / 做集成。

### 1.2 全量训练里，验证集到底干什么

只有两件事，**都只需要掩膜**：

1. **选 best**：`checkpoint_metric: val_dice_peri`（`configs/train.yaml`），按每通道阈值搜索后的 Dice；
2. **写阈值进 checkpoint**：验证集上扫出的 `thresholds` 会随权重存盘，推理直接使用（避免固定 0.5 的损失）。

所以：

| 验证集里的东西 | 全量训练需要吗 |
|---|---|
| 影像（`*.nii.gz`） | **要**（要过模型算 Dice） |
| 掩膜（`*_mask.nii.gz`） | **要**。没有它 → `03_train.sh full` 当场报错 |
| `SeriesType.xlsx` | 要（决定掩膜归 core 还是 peri） |
| 结构化字段表（病理/WHO 分级） | **不需要**，val 里不参与任何计算 |
| `fake/`、`compositing/`、`duplicate/` 金标准 | 不需要，目标是"能算的才算"，算不了会明确打印跳过 |

### 1.3 三类数据要不要准备

| 数据 | 必需性 | 用途 |
|---|---|---|
| **训练集** `training/` | **必需** | 全量的 train 全来自它 |
| **验证集** `verification/` | **全量训练的必需项** | 早停 + 阈值 + 最终指标口径 |
| 评测集 `evaluation_*` | 不用准备 | 评测时平台通过 `input.dataset_path` 下发 |

---

## 2. 数据准备

### 2.1 目录布局（训练集与验证集同构，只是容器名不同）

```text
DATASET_ROOT=/2026aicompetition/datasets/training
└── annotation/                                        ← 容器：训练集叫 annotation
    ├── SeriesType.xlsx                                ★ 模态表（AccessionNumber / SeriesUid / SeriesType）
    ├── 脑胶质瘤标注结果-训练集.xlsx                       ★ 字段金标准（只有训练集有）
    ├── <32位检查号>/
    │   ├── <2.25.* 序列UID>.nii.gz                     ← 影像
    │   └── <2.25.* 序列UID>_<RoiName>_<RoiNumber>_mask.nii.gz   ← 掩膜（与影像同目录！）
    └── {fake, compositing, duplicate}/                 特殊影像 / 重复影像金标准

VAL_ROOT=/2026aicompetition/datasets/verification
└── original/                                          ← 容器：验证集叫 original
    ├── SeriesType.xlsx                                ★ 有（模态靠它）
    └── <检查号>/ …                                     ← 影像 + 掩膜，同上；**没有字段金标准表**
```

要点：

- **影像与掩膜平铺在同一层**，只靠文件名区分；掩膜必须能按 `_mask` 后缀认出来。
- 数据根填**上一层**会被自动下钻并告警（`…/training` 或 `…/training/annotation` 都能跑通）。
- **别把掩膜漏搬**：只搬影像的话，验证集会"有图没答案"，全量直接跑不起来（见 §6）。

### 2.2 环境变量与路径

| 变量 | 作用 | 备注 |
|---|---|---|
| `DATASET_ROOT` | 训练集根 | 覆盖 `configs/paths.yaml` 的 `raw.track4` |
| `VAL_ROOT` | 验证集根 | 覆盖 `raw.val`；**填了才算启用 external 口径** |
| `WORKSPACE` | 平台私有存储 | 默认 `/2026aicompetition/workspace`；日志/答案/导出权重都在这下面 |
| `CACHE_DIR` | 预处理缓存目录 | 默认 `data/preprocess_cache` |
| `GLIOMA_LABELS_DIR` | 表的兜底搜索目录 | 表在非常规位置时才需要 |

### 2.3 验证集的掩膜从哪来（**不是**从 xlsx 来）

掩膜是**数据集里的文件**，不是表里的列：

| 内容 | 载体 | 谁读 |
|---|---|---|
| 影像 | `<检查号>/<序列UID>.nii.gz` | 探针扫盘 |
| **掩膜** | `<检查号>/<序列UID>_<RoiName>_<RoiNumber>_mask.nii.gz` | 探针按 `_mask` 后缀扫盘 |
| 模态（把掩膜归到 core / peri） | `SeriesType.xlsx` | `read_series_types` |
| 字段（病理结果 / WHO 分级） | `脑胶质瘤标注结果-*.xlsx` 的 `检查级别` sheet | `read_structured_table` |

`RoiName` / `RoiNumber`（`瘤体`/`水肿`/`肿瘤瘤体`…）在那张表的 `ROI级别` sheet 里确实列了，但代码只解析**文件名**、不读那两列。

**在真数据上一眼确认（三条，任一为 0 就说明验证集没有掩膜）：**

```bash
find "$VAL_ROOT" -name '*_mask.nii.gz' | wc -l          # ① 掩膜文件到底有没有
bash scripts/01_probe.sh --val                          # ② 看报告 mask_role_counts
python -c "import json;r=json.load(open('data/manifest_val.json',encoding='utf-8'));\
print(r['report']['mask_role_counts'], sum(1 for c in r['cases'] if c.get('masks')))"
```

`mask_role_counts` 形如 `{'core': 120, 'peri': 118}` 就是拿到了。若为 `{}`：验证集这一路没有早停信号 → 改用折内 val（`bash scripts/03_train.sh 0`），见 §6。

### 2.4 数据体检：探针报告要看哪几个数字

`bash scripts/01_probe.sh`（训练集）与 `--val`（验证集）各跑一次，全部报告都会打印到 stdout，并写进对应清单的 `report` 键。

| 字段 | 期望 | 不正常时怎么办 |
|---|---|---|
| `n_cases` | 与平台公布的例数一致（缺几例看下一行） | `cases_dropped_no_series` 有值 → 那些检查一路序列都没认出模态 |
| `series_type_rows` | > 0 | = 0 → 类型表没读到；看 `series_type_key_hits`（检查号/序列号命中数）区分"串表"还是"键口径不一致" |
| `series_desc_rows` | 辅助 | 类型表拿不到时靠标注表的「序列描述」兜底判模态（探针会打印"已用…旁证"） |
| `mask_role_counts` | `core` / `peri` 都有 | 只有 `core` → 验证集/训练集缺 FLAIR/T2；为空 → 见 §2.3 |
| `modality_counts` | `t1c`/`flair`/`t2` 齐 | 缺哪个看 `missing_t1c` / `missing_flair`（缺失会自动降级用 fallback 顶替） |
| `cases_without_input_channel` | 了解数量即可 | 整例序列被类型表标成 `其他`（或只有 DWI/ADC/SWI）→ **照常进训练**（4 通道全零 + 借该例几何），但这类例回传近噪声梯度，验证集里若有会拖低 Dice |
| `unknown_series_total` | 0 最好 | > 0 → 那些序列模态没认出来，只能靠体素判别模型（`scripts/31_train_modality_model.py`） |
| `special.gold_pairs` | > 0 最好 | = 0 → 重复影像金标准没解析出来（看 `gold_files`），该指标将算不了 |

保存的 `mask_role_counts` 与掩膜判据都依赖**文件名后缀**，所以搬数据时**必须连掩膜一起搬**。

---

## 3. 训练

### 3.1 前置检查

```bash
# 清单必须在，且是**当前数据根**产出的（换过数据根就必须重跑 01_probe）
python -c "import json;d=json.load(open('data/manifest.json',encoding='utf-8'));\
print('source=',d.get('data_source'),'root=',d.get('data_root'),'cases=',len(d['cases']))"

# 验证集清单必须在（否则 full 直接报错）
python -c "import json;print(len(json.load(open('data/manifest_val.json',encoding='utf-8'))['cases']))"
```

### 3.2 命令

```bash
bash scripts/03_train.sh full
# 等价于：python -m src.training.trainer --config train --fold full --tag g4_full
```

常用变体：

| 目的 | 命令 |
|---|---|
| 换配置（如改 `epochs`/`batch_size`） | `CONFIG=train_v2 bash scripts/03_train.sh full`（产物目录仍是 `g4_full`） |
| 多 seed（集成用） | `sed 's/^seed: 42/seed: 43/' configs/train.yaml > configs/train_s43.yaml`，然后 `CONFIG=train_s43 TAG_PREFIX=g4_full43 bash scripts/03_train.sh full` → 目录为 `checkpoints/g4_full43_full/` |
| 从某个权重继续 | `PRETRAINED=checkpoints/g4_fold0/best.pth bash scripts/03_train.sh full` |
| 中断后续训 | 重跑同一条命令即可（自动读 `last.pth`；`--no-resume` 可关） |
| 先建缓存（推荐） | `python scripts/13_build_cache.py --root "$DATASET_ROOT" --workers 8`（首次较慢，之后每例 2.2s → 0.2s 量级） |

**全量模式不需要 `02_build_dataset.sh`**（它只建 `folds.json`，全量不读）。跑过也没关系。

### 3.3 会打印什么、写哪些文件

启动时会打印：

```text
[trainer] 数据源标识: train=official/... val=official/...
[trainer] 全量模式：train=N（清单全部病例） val=M（官方验证集 data/manifest_val.json；无掩膜的例已剔除——选模只用 Dice）
[trainer] tag=g4_full fold=full train=N val=M steps/epoch=… 特殊影像正样本=[a, b] 重复对=True
```

| 产物 | 说明 |
|---|---|
| `checkpoints/g4_full/best.pth` | **选模指标最优**的权重（下游评估/导出用它） |
| `checkpoints/g4_full/last.pth` | 最近一个 epoch（断点续训用） |
| `logs/train_full.log` | 训练 stdout 全量（`03_train.sh` 内 `tee`） |
| `logs/g4_full.jsonl` | 规范 JSONL 日志（`timestamp/epoch/step/phase/loss/lr/data_source`） |
| `data/preprocess_cache/` | 若建过缓存 |

### 3.4 选模与阈值（全量特有的三点）

1. **选模只看 `val_dice_peri`**：验证集里**没有掩膜**的病例会被剔除并打印，绝不混进 Dice（否则"预测为空 + 空 target"会按 `den == 0` 记成 **1.0 的假满分**，把 best 选歪）。
2. **阈值随权重存**：验证集上按通道扫出的 `thresholds` 写进 `best.pth`；`16_finalize.sh` 的 1/5 步会用**同口径**再标定一次并写回所有折（保证"标定口径 == 评估口径"）。
3. **验证集里"没有真输入通道"的病例**（整例序列被标成 `其他`）：按"全放开"口径照算，但输入是全零 → 预测基本为空 → Dice 基本恒 0，会**往下拖**选模指标。trainer 会打印这类例数；在意就从验证集清单里去掉它们。

---

## 4. 验证与评估

### 4.1 一键收尾（推荐）

```bash
FOLDS=full bash scripts/16_finalize.sh
# 只想看"用哪些权重评哪些病例"（不跑推理）：
FOLDS=full bash scripts/16_finalize.sh --print-split      # → split=external / external: g4_full
```

它会依次做 5 步（`docs/TRACK4_RUNBOOK.md` 有逐步解释）：

| 步骤 | 做什么 | 产物 |
|---|---|---|
| 1/5 | 集成阈值标定（**external 口径**，写回权重） | — |
| 2/5 | 全图评估：验证集 Dice / NSD / HD95 | `logs/eval_external_ens.log` |
| 3/5 | 目标一/二 + 重复影像 | stdout（`tail -30`） |
| 4/5 | 按规范 §5.2 导出提交权重 | `{WORKSPACE}/checkpoint/…` |
| 5/5 | Mock Competition + 提交前检查 | — |

`FOLDS` 可以写折号（`"0 1 2"`）或完整 tag（`full` / `g4_full43_full`）。

### 4.2 分步（想知道每一步在干什么时）

```bash
# ① 阈值标定（必须用**同一批权重**：集成模型就整批一起标）
python scripts/14_calibrate_thresholds.py --split external \
    --ckpt checkpoints/g4_full/best.pth --write

# ② 分割指标（验证集 = 最终口径）
CKPT=checkpoints/g4_full/best.pth bash scripts/04_eval.sh --split external

# ③ 目标一/二 + 重复影像
python scripts/15_eval_special_dup.py --split external --ckpt checkpoints/g4_full/best.pth
```

⚠️ **`--ckpt` 一定要给**：自动发现只认 `checkpoints/g4_fold*/`，不给就会拿折权重去评（数字对不上时先查这里）。

### 4.3 全量 + 验证集口径下，哪些指标算得了

| 指标 | 全量（external） | 说明 |
|---|---|---|
| 分割 Dice / NSD / HD95 | **算得了** | 只统计**有掩膜**的验证集病例；`n_no_mask` 是跳过数 |
| 目标一（假人体）/ 目标二（拼接） | 看验证集 | 验证集自带 `fake/`、`compositing/` 才算得了；没有会明确打印"跳过" |
| 重复影像 | 看验证集 | 需要验证集自己的 `duplicate/` 金标准；没有 → 打印"跳过重复影像指标" |
| 结构化字段 / WHO 分级 | **算不了** | 验证集**没有**字段金标准表；要量就用**折内 OOF**（`04_eval.sh --split fold`） |

一句话：**分割看验证集；字段看折内。**

### 4.4 结果在哪

- 分割（external）：`logs/eval_external_ens.log`（`16_finalize.sh` 2/5 的输出）
- 目标一/二 + 重复：`16_finalize.sh` 3/5 的 stdout
- 规范 JSONL：`logs/g4_full.jsonl`（训练）、评估脚本各自的 `.log`
- **不要**把"单权重评验证集"的数字当最终口径：验证集与训练集无交集，**集成**（多 seed / 多折）才是最终口径。

---

## 5. 交付

```bash
# 导出全量权重到平台约定目录（{WORKSPACE}/checkpoint/<goal>/*.pt）
bash scripts/09_export_submission.sh full          # 也可写 g4_full / g4_full43_full

# 复核导出结果
bash scripts/09_export_submission.sh --verify

# 提交前检查（本机必然有几项是"容器内步骤"，正常）
bash scripts/23_pre_submit_check.sh
```

要点：

- `09` 会自动归一化：`full → g4_full`；只给折号 `0` 会补成 `g4_fold0`。
- 单权重时 `goal5_segmentation/` 下是 `core.pt`；多权重时保留 `<tag>.pt` 供集成。
- 推理侧不设 `GLIOMA_CKPT` 时会自动读取该目录下**全部** `.pt` 做集成。
- 链接（硬链接）还是拷贝：`09` 默认 `copy`（跨文件系统/打包上传更稳）。
- ⚠️ 别把 1~2 epoch 的冒烟权重留在 `{WORKSPACE}/checkpoint/` 里 —— `--verify` 只查文件齐不齐，查不出"训没训过"。

---

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `03_train.sh full` 报"需要官方验证集 / 验证集清单里**没有一例带掩膜**" | 没配 `VAL_ROOT` 或没跑 `--val`；或验证集真的没掩膜 | 配 `VAL_ROOT` + `01_probe.sh --val`；确实没掩膜 → 用折内 val：`bash scripts/03_train.sh 0` |
| 探针报 `series_type_rows=0`、模态全 `other` | 类型表没读到 | `python scripts/30_inspect_table.py --root $DATASET_ROOT --rows 3` 看表和列名；表在非常规位置时 `export GLIOMA_LABELS_DIR=<目录>` |
| 探针报 `mask_role_counts={}` | 掩膜被漏搬 / 命名不符 | `find <根> -name '*_mask.nii.gz' \| head`；掩膜必须与影像同目录、以 `_mask` 结尾 |
| 评估数字异常（低得离谱） | 用了折权重评验证集 | 显式 `--ckpt checkpoints/g4_full/best.pth` |
| 明明有验证集，`16_finalize.sh` 却说"未接入"、评估走 OOF | 探针异常被吞掉（典型：**GBK 控制台**下打印 ⚠️ 抛 `UnicodeEncodeError`） | 已内置编码兜底；若再遇到，脚本会把探针原文打出来（`⚠️ 验证集探针没有产出结果`），按原文修 |
| `09_export_submission.sh` 说"未找到 g4_fold*/best.pth" | 只训了全量 | 传 tag：`bash scripts/09_export_submission.sh full` |
| 中断后想接着训 | — | 重跑同一条命令（读 `last.pth`）；换数据必须 `13_build_cache.py --force` + 重跑 `01_probe.sh` |
| 显存不足 | — | `batch_size` 2→1 → `patch_size` `96³`→`80³` → 关 TTA（`seg_tta_flips: []`）→ `num_workers` → 2 |

---

## 7. 附录

### 7.1 命令速查

| 步骤 | 命令 |
|---|---|
| 训练集探针 | `bash scripts/01_probe.sh` |
| 验证集探针 | `export VAL_ROOT=… && bash scripts/01_probe.sh --val` |
| （可选）缓存 | `python scripts/13_build_cache.py --root "$DATASET_ROOT" --workers 8` |
| 全量训练 | `bash scripts/03_train.sh full` |
| 收尾（一键） | `FOLDS=full bash scripts/16_finalize.sh` |
| 阈值标定 | `python scripts/14_calibrate_thresholds.py --split external --ckpt checkpoints/g4_full/best.pth --write` |
| 分割评估 | `CKPT=checkpoints/g4_full/best.pth bash scripts/04_eval.sh --split external` |
| 目标一二/重复 | `python scripts/15_eval_special_dup.py --split external --ckpt checkpoints/g4_full/best.pth` |
| 导出权重 | `bash scripts/09_export_submission.sh full` |
| 提交前检查 | `bash scripts/23_pre_submit_check.sh` |
| 划分口径自检 | `FOLDS=full bash scripts/16_finalize.sh --print-split` |

### 7.2 验收清单

数据：

- [ ] `data/manifest.json` 的 `data_source`/`data_root` 指向**当前**数据根，`n_cases` 合理
- [ ] `data/manifest_val.json` 存在，且 `mask_role_counts` 非空（有 core/peri）
- [ ] 掩膜与影像**同目录**、命名以 `_mask.nii.gz` 结尾（`find … -name '*_mask.nii.gz' | wc -l` 非 0）
- [ ] 验证集**没有**字段金标准表是**预期**（不必去找）

训练：

- [ ] 启动日志出现 `[trainer] 全量模式：train=… val=…`
- [ ] `checkpoints/g4_full/best.pth` 与 `last.pth` 都在
- [ ] `logs/train_full.log` 与 `logs/g4_full.jsonl` 在写

评估 / 交付：

- [ ] `FOLDS=full bash scripts/16_finalize.sh --print-split` 输出 `split=external` 且列出 `g4_full`
- [ ] `logs/eval_external_ens.log` 里有分割指标与 `n_no_mask`（跳过数）
- [ ] `bash scripts/09_export_submission.sh --verify` 六个 goal 全 `ok`
- [ ] `bash scripts/23_pre_submit_check.sh` 通过（本机允许"权重未导出"这类容器内步骤）
- [ ] `{WORKSPACE}/checkpoint/` 里**没有**冒烟权重
