# 路线 A · 一体化训练 + 提交 完整 README

> **本文是自包含的**：从代码上传、环境准备、数据核查，到全量训练、评估导出、提交自检，
> 只看这一份就能跑完全流程（另可与 `FULL_TRAIN_GUIDE.md` 互参：本文是它的**超集**，
> 多出"六个模型的交付与加载机制"、路线 B 兜底、报错对照与命令速查）。
>
> 工程：`glioma_track4`（训练主力）＋ `Glioma_recognition-main`（提交）＋ `glioma_goals`（可选兜底，见附录 A）
> 入口：`scripts/03_train.sh`　|　全量配置：`configs/train20.yaml`

---

## 0. 路线 A 是什么

**一个多任务骨干，一次前向同时产出六路输出**，每个 optimization step 六个目标一起前进：

| 损失项（`configs/train.yaml` → `loss`） | 对应目标 |
|---|---|
| `dice` + `ce` + `deep_sup` | ⑤ 分割（core / peri 两通道 + 深监督） |
| `cls` | ③ 病灶识别（`TumorProbability`）+ ④ 14 个结构化字段 |
| `special` | ① 影像真实性（假人体）+ ②-A 拼接 |
| `embed` | ②-B 重复影像（配对嵌入） |
| `contain` | 集合约束 core ⊆ peri |

所以**不存在"goal1 训完再训 goal2"** —— 六个目标天然同步推进，每一步都在同时学六路。

### 0.1 "一个骨干" ≠ "一个模型文件"（重要）

交付结构仍然是**六个目录、六个模型文件**（规范 §5.2）：

```text
/2026aicompetition/workspace/checkpoint/
├── goal1_authenticity/model.pt
├── goal2_stitched/model.pt
├── goal2_duplicate/encoder.pt
├── goal3_tumor/model.pt
├── goal4_diagnosis/model.pt
└── goal5_segmentation/core.pt (+flair.pt)
```

`09_export_submission.sh` 把**同一份多任务权重复制到这六个路径**：

```56:66:glioma_track4/scripts/09_export_submission.sh
  case "$g" in
    goal2_duplicate) names="encoder.pt" ;;
    goal5_segmentation) names="core.pt flair.pt" ;;
    *) names="model.pt" ;;
  esac

  for n in $names; do
    cp -f "$src" "$dst_dir/$n"
  done
```

"一个骨干"只意味着**这六份文件的内容同源**；对提交侧来说依然是六个独立模型。

### 0.2 六个任务分别加载，为什么不会报错

**① 每个任务有自己的权重路径**（提交工程 `tasks/<goal>/config.py`）：

```24:24:Glioma_recognition-main/tasks/goal1_authenticity/config.py
    ckpt_rel: str = "goal1_authenticity/model.pt"
```
```9:9:Glioma_recognition-main/tasks/goal4_diagnosis/config.py
    ckpt_rel: str = "goal4_diagnosis/model.pt"
```

**② 各自独立 load + 独立校验**：`BackboneRunner.load()` 对每个 `ckpt_rel` 单独 `torch.load`，
用**该 checkpoint 自己记录的** `model_cfg` / `cls_spec` 重建骨干，再跑核心层完整性 +
`_check_heads` + `cls_spec ⇄ cls_heads` 一致性校验。六个任务互不干扰。

**③ 三份骨干实现的头定义逐行一致**（`glioma_goals/shared/backbone_mednext.py`、
`glioma_track4/src/models/mednext.py`、`Glioma_recognition-main/tasks/_common/mednext.py`）：

```199:205:Glioma_recognition-main/tasks/_common/mednext.py
        feat = ci + chs[-1]
        self.cls_heads = nn.ModuleList([nn.Linear(feat, n) for _n, n in cls_spec])
        self.field_names = [n for n, _ in cls_spec]
        self.embed_head = nn.Sequential(nn.Linear(feat, embed_dim), nn.GELU(),
                                        nn.Linear(embed_dim, embed_dim))
        self.special_head = nn.Sequential(nn.Linear(feat, 64), nn.GELU(),
                                          nn.Linear(64, 2))
```

参数名与形状完全对得上（`embed_dim=128` 默认值、`pooled = cat([pool(h), pool(bottleneck)])` 也一致）。

> `_check_heads` 只在"权重里**根本没有** `special_head.*` / `embed_head.*` 参数"时才抛错
> （`tasks/_common/backbone_runner.py:76-95`）。路线 A 的权重这两组头都齐，必然通过。

**④ 那条必须避开的红线**：不要把六个任务都指向**同一份单任务权重**。例如让目标④也去读
`goal1_authenticity/model.pt` —— 那份权重里 `cls_heads` 只有 1 个头、`embed_head` 从未被监督，
结果是**不报任何错**，但 14 个字段与嵌入全是随机输出，指标莫名其妙地差。
路线 A 的权重不存在这个问题（六路都被真正训练过）。

### 0.3 与路线 B 的区别

| | 路线 A（本文） | 路线 B（附录 A） |
|---|---|---|
| 组织 | 1 个多任务模型 | 6 个独立单任务模型 |
| 折支持 | 折训练 + **全量训练**都有 | 只有折 / `val_ratio` |
| 断点续训 | ✅ 默认开启 | ❌ 无 |
| 阈值标定 / 评估 / 导出 | `16_finalize.sh` 一条龙 | 需自己跑 `evaluate.py` |
| 六份权重的关系 | 内容同源 | 各自独立训练 |

**两条路线交付的都是六个模型文件**，只是训练方式不同。

### 0.4 全量 vs 折内交叉验证

| | 折内 CV（`03_train.sh 0` / `all`） | **全量（`03_train.sh full`）** |
|---|---|---|
| train | 清单里 `1 - val_ratio` 的病例 | **清单全部病例** |
| val | 同一份清单的留出折 | **官方验证集**（与训练集无交集） |
| 选 best | 折内 val Dice | 验证集 Dice |
| 权重目录 | `checkpoints/g4_fold0/` … | `checkpoints/g4_full/` |
| 需要 `folds.json` | 需要（`02_build_dataset.sh`） | **不需要**（全量不读它） |
| 用途 | 出 OOF、量"没有验证集金标准"的指标 | **最终交付**（数据一点不浪费） |
| 代价 | 每折少用 20% 数据 | 没有 OOF（选不出"没见过这例"的模型） |

两者**不冲突**：常见做法是**全量出交付权重，折内模型用来量指标 / 做集成**。

**三条铁律**

1. **没有验证集就不要走全量** —— `03_train.sh full` 会直接报错，不会静默拿训练集当验证集。
2. **验证集必须有掩膜**（`*_mask.nii.gz`）；字段标注表（`脑胶质瘤标注结果-*.xlsx`）**不需要**。
3. 全量权重在 `checkpoints/g4_full/`，而**自动发现只认 `g4_fold*`** —— 评估/独立脚本要显式给
   `--ckpt` 或 `FOLDS=full`（`09_export_submission.sh` 例外，它内置了 `g4_full` 兜底）。

---

## 1. 上传后的目录落位

三个工程必须**并排**放在同一个父目录下（脚本里写死了相对关系）：

```text
/2026aicompetition/workspace/dcs/
├── glioma_track4/             ← 算法工程：数据管线 + 一体化训练 + 全部脚本（主力）
├── glioma_goals/              ← 训练工程：六目标独立训练（附录 A 兜底用）
├── Glioma_recognition-main/   ← 提交工程：镜像内推理服务
└── CLOUD_USAGE_GUIDE.md       ← 另一份流程文档（本文已自包含，可不看）
```

```bash
cd /2026aicompetition/workspace/dcs
ls -d */                                                  # 期望看到三个工程目录
chmod +x glioma_track4/scripts/*.sh Glioma_recognition-main/start.sh 2>/dev/null
```

**平台路径约定**（不要改）：

| 用途 | 路径 |
|---|---|
| 训练集 | `/2026aicompetition/datasets/training/annotation` |
| 官方验证集 | `/2026aicompetition/datasets/verification`（其下是 `original/`） |
| 提交权重 | `/2026aicompetition/workspace/checkpoint/<goal>/` |
| 答案 | `/2026aicompetition/workspace/answer/<evaluation_id>/` |
| 日志 | `/2026aicompetition/workspace/logs` |

> 数据集不是天然可见的：创建容器时要在「存储与数据服务」里勾选训练/验证影像数据集，
> 没勾选的话目录压根不存在。

---

## 2. 环境准备

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

# 平台容器自带 torch → 用 system 模式，只补轻量包（nibabel/openpyxl/fastapi…）
bash scripts/00_setup_env.sh --mode system

# 云桌面禁网：用离线 wheel 目录
# bash scripts/00_setup_env.sh --mode system --offline --wheels wheels
# 只体检不安装
# bash scripts/00_setup_env.sh --mode verify
```

自检期望：

```text
[check] python 3.x | torch 2.x | cuda 12.x | 可用 True
[check] 缺失依赖: 无 ✓
```

> 报 `Error: externally-managed-environment` 不用手工处理，脚本会自动加 `--break-system-packages`。
> 容器里往往只有 `python3`：直接跑 `.py` 用 `python3`，`.sh` 脚本会自己探测解释器。

---

## 3. 数据根定位

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
python3 scripts/29_locate_dataset_root.py
```

末行直接给结论：

```text
结论：数据根 = /2026aicompetition/datasets/training/annotation（3255 例）
      export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
```

`.../training` 或 `.../training/annotation` **都能跑通**（前者自动下钻）。
唯一会被直接拦下的是 `/2026aicompetition/datasets`（多阶段父目录 → `ValueError`，这是预期行为）。

---

## 4. 数据布局与探针

### 4.1 目录布局（训练集与验证集同构，容器名不同）

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

- **影像与掩膜平铺在同一层**，只靠文件名区分（掩膜必须以 `_mask` 结尾）。
- **搬数据必须连掩膜一起搬**：只搬影像的话验证集会"有图没答案"，全量直接跑不起来。
- 掩膜的 core / peri 归属由 `SeriesType.xlsx` 决定（FLAIR/T2 上的"瘤体"算 peri 而不是 core）。

### 4.2 训练集探针

```bash
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
bash scripts/01_probe.sh          # → data/manifest.json
```

### 4.3 探针报告要看哪几个数字

| 字段 | 期望 | 不正常时怎么办 |
|---|---|---|
| `n_cases` | 与平台公布例数一致 | `cases_dropped_no_series` 有值 → 那些检查一路序列都没认出模态 |
| `series_type_rows` | **> 0** | = 0 → 类型表没读到；看 `series_type_key_hits` 区分"串表"还是"键口径不一致" |
| `modality_counts` | `t1c` / `flair` / `t2` / `t1` 齐 | 缺哪个看 `missing_t1c` / `missing_flair`（缺失会自动降级用 fallback 顶替） |
| `mask_role_counts` | `core` / `peri` 都有 | 只有 `core` → 缺 FLAIR/T2；为空 → 掩膜漏搬或命名不符（见 §5.2） |
| `label_field_counts` | 越多越好 | 0 → 结构化字段金标准没读到，目标③④ 学不到东西 |
| `cases_without_input_channel` | 了解数量即可 | 整例序列被标成 `其他`（或只有 DWI/ADC/SWI）→ **照常进训练**（4 通道全零 + 借该例几何），但回传近噪声梯度 |
| `unknown_series_total` | 0 最好 | > 0 → 可用 `python3 scripts/31_train_modality_model.py --root <数据根>` 训体素判别兜底 |
| `special.gold_pairs` | > 0 最好 | = 0 → 重复影像金标准没解析出来，该指标算不了 |

数据信息表就在数据集里（与病例目录同层），探针自动读、零配置。
真找不到时用 `export GLIOMA_LABELS_DIR=<含表的目录>` 显式指定。

---

## 5. 官方验证集清单（`full` 模式的 val）

### 5.1 它在全量训练里到底干什么

只有两件事，**都只需要掩膜**：

1. **选 best**：`checkpoint_metric: val_dice_peri`（按每通道阈值搜索后的 Dice）；
2. **把阈值写进 checkpoint**：验证集上扫出的 `thresholds` 随权重存盘，推理直接使用
   （避免固定 0.5 带来的掉点）。

所以验证集里**需要什么 / 不需要什么**：

| 验证集里的东西 | 全量训练需要吗 |
|---|---|
| 影像 `*.nii.gz` | **要**（要过模型算 Dice） |
| 掩膜 `*_mask.nii.gz` | **要**。没有它 → `03_train.sh full` 当场报错 |
| `SeriesType.xlsx` | 要（决定掩膜归 core 还是 peri） |
| 结构化字段表（病理/WHO 分级） | **不需要**，val 里不参与任何计算 |
| `fake/`、`compositing/`、`duplicate/` 金标准 | 不需要（有才算，没有会明确打印"跳过"） |

### 5.2 掩膜从哪来 + 三步确认

掩膜是**数据集里的文件**，不是表里的列：

```bash
export VAL_ROOT=/2026aicompetition/datasets/verification

find "$VAL_ROOT" -name '*_mask.nii.gz' | wc -l      # ① 掩膜文件到底有没有
bash scripts/01_probe.sh --val                      # ② → data/manifest_val.json【全量训练的硬前置】
python3 - <<'PY'                                    # ③ 看判据（下面 §5.3）
import json, os
m = json.load(open("data/manifest_val.json", encoding="utf-8"))
print("mask_role_counts =", m["report"]["mask_role_counts"])
print("病例:", len(m["cases"]), "| 带掩膜:", sum(1 for c in m["cases"] if c.get("masks")))
PY
```

`mask_role_counts` 形如 `{'core': 120, 'peri': 118}` 就是拿到了。
若为 `{}` 或 `带掩膜 == 0`：**这条线没有早停信号** → 改用折内 val（`bash scripts/03_train.sh 0`），
或让数据方补验证集掩膜。

> 另外：官方验证集实测**只有 `SeriesType.xlsx`、没有字段金标准表**，这是预期的，
> 不要因为 `label_field_counts` 为空去满磁盘找表。

### 5.3 硬前提自检

全量模式的 val 只用带掩膜的病例算 Dice 来选 best，一例都没有时训练器会直接退出
（`src/training/trainer.py:203-217`）。上面的第 ③ 步 `带掩膜 > 0` 就是放行条件。

> `configs/paths.yaml` 里 `raw.val` 已按平台路径填好，不 export `VAL_ROOT` 也能跑；
> 想让评估回退折内 val：把 `raw.val` 清空，或删掉 `data/manifest_val.json`。

---

## 6. 预处理缓存（可选，强烈建议）

```bash
export CACHE_DIR=/2026aicompetition/workspace/cache      # 私有存储，容器删了不丢
python3 scripts/13_build_cache.py --root "$DATASET_ROOT" --workers 8
```

把 1mm 公共网格 + 整脑视图的重采样开销提前做掉，训练时直接命中缓存
（首次较慢，之后每例从 2.2s 量级降到 0.2s 量级）。

---

## 7. 冒烟：先证明链路通（几分钟，必须做）

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

# ① 合成自检（不需要真实数据、不需要 GPU）
python3 scripts/99_smoke_test.py

# ② 全量路径的 2-epoch 冒烟（小模型 + 小 patch，产物落 checkpoints/smoke_full/）
CONFIG=_smoke_train TAG_PREFIX=smoke bash scripts/03_train.sh full
```

期望日志开头：

```text
[trainer] tag=smoke_full fold=full train=<清单全部病例> val=<带掩膜验证例数> steps/epoch=...
[trainer] 全量模式：train=xxxx（清单全部病例） val=xxx（官方验证集 .../manifest_val.json）
```

2 个 epoch 结束、`checkpoints/smoke_full/best.pth` 出现 = 链路通过。

> **冒烟权重绝不能留在提交路径**：`TAG_PREFIX=smoke` 让它落在 `checkpoints/smoke_full/`，
> `09_export_submission.sh` 的自动选择只挑 `g4_fold*` / `g4_full*`，不会误选它；
> 用完可 `rm -rf checkpoints/smoke_full`。

---

## 8. 训练

### 8.1 三种模式

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

# ① 单折（前台，产物 checkpoints/g4_fold0/）
bash scripts/03_train.sh 0

# ② 多折并行（4 张卡 → GPU0..GPU3；第 5 折后补 bash scripts/03_train.sh 4）
bash scripts/03_train.sh all 4

# ③ 全量训练（train=清单全部病例，val=官方验证集，无折划分）★ 本方案用这个
bash scripts/03_train.sh full
```

| 模式 | tag（→ `checkpoints/<tag>/`） | 日志 |
|---|---|---|
| 单折 | `g4_fold<N>` | `logs/train_fold<N>.log` |
| 多折并行 | `g4_fold0..` | `logs/train_fold<N>.log`（各折一份） |
| 全量 | `g4_full` | `logs/train_full.log` |

环境变量：`CONFIG`（用哪份 `configs/*.yaml`）、`TAG_PREFIX`（产物前缀，默认 `g4`）、
`PRETRAINED`（加载预训练权重）、`FOLD`（单折折号）、`PY`（解释器）。

### 8.2 正式：全量 20 epoch

`epochs` 写在**配置文件**里（`03_train.sh` 只透传 `--config`，没有 `--epochs` 参数）。
工程已备好 `configs/train20.yaml`（与 `train.yaml` 唯一差异：`epochs: 20`）：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
export VAL_ROOT=/2026aicompetition/datasets/verification
export CACHE_DIR=/2026aicompetition/workspace/cache
export WORKSPACE=/2026aicompetition/workspace

CONFIG=train20 nohup bash scripts/03_train.sh full > logs/full20_launch.log 2>&1 &
tail -f logs/train_full.log
```

启动成功会看到：

```text
[trainer] 数据源标识: train=official/... val=official/...
[trainer] tag=g4_full fold=full train=N val=M steps/epoch=… 特殊影像正样本=[a, b] 重复对=True
[trainer] 全量模式：train=xxxx（清单全部病例） val=xxx（官方验证集 .../manifest_val.json；无掩膜的例已剔除——选模只用 Dice）
```

**单卡注意事项**：不要设 `CUDA_VISIBLE_DEVICES=1`（单卡机器上会把唯一的卡隐藏掉，
`torch.cuda.is_available()` 变 False）。留空（默认 0）或显式 `=0`。

**想先跑 5 轮快速看效果**（OneCycle 会在 5 轮内完整退火，属于"正经训完"的短模型）：

```bash
sed 's/^epochs: 20/epochs: 5/' configs/train20.yaml > configs/train5.yaml
CONFIG=train5 bash scripts/03_train.sh full
```

### 8.3 常用变体

| 目的 | 命令 |
|---|---|
| 换配置（改 `epochs` / `batch_size`） | `CONFIG=train_v2 bash scripts/03_train.sh full`（产物目录仍是 `g4_full`） |
| **多 seed（集成用）** | `sed 's/^seed: 42/seed: 43/' configs/train.yaml > configs/train_s43.yaml`，然后 `CONFIG=train_s43 TAG_PREFIX=g4_full43 bash scripts/03_train.sh full` → `checkpoints/g4_full43_full/` |
| 从某个权重继续 | `PRETRAINED=checkpoints/g4_fold0/best.pth bash scripts/03_train.sh full` |
| 中断后续训 | 重跑同一条命令即可（自动读 `last.pth`） |
| 无视断点、从零重训 | `python3 -m src.training.trainer --config train20 --fold full --tag g4_full --no-resume` |

> `03_train.sh` 不转发 `--no-resume`，所以要"从零"必须用上表最后那条底层命令。
> **全量模式不需要 `02_build_dataset.sh`**（它只建 `folds.json`，全量不读）。

### 8.4 选模与阈值（全量特有的三点）

1. **选模只看 `val_dice_peri`**：验证集里**没有掩膜**的病例会被剔除并打印，绝不混进 Dice ——
   否则"预测为空 + 空 target"会按 `den == 0` 记成 **1.0 的假满分**，把 best 直接选歪。
2. **阈值随权重存**：验证集上按通道扫出的 `thresholds` 写进 `best.pth`；
   `16_finalize.sh` 的 1/5 步会用**同口径**再标定一次并写回（保证"标定口径 == 评估口径"）。
3. **验证集里"没有真输入通道"的病例**（整例序列被标成 `其他`）：按"全放开"口径照算，
   但输入全零 → 预测基本为空 → Dice 基本恒 0，会**往下拖**选模指标。trainer 会打印这类例数；
   在意就从验证集清单里去掉它们。

---

## 9. 配置旋钮（`configs/train20.yaml`）

| 键 | 默认 | 说明 / 什么时候改 |
|---|---|---|
| `epochs` | 20 | 总轮数。**OneCycle 的学习率退火跨度 = 这里**，要设成你真正打算跑完的轮数 |
| `batch_size` | 2 | 显存不够降到 1 |
| `patch_size` | `[96,96,96]` | 显存不够降到 `[80,80,80]` |
| `num_workers` | 4 | CPU 核多可到 8 |
| `val_every` | 1 | 每 N 轮验证一次。**建议保持 1**（见 §11 陷阱 2） |
| `lr` / `weight_decay` | 3e-4 / 3e-5 | `OneCycleLR(max_lr=lr, total_steps=epochs×steps_per_epoch)` |
| `grad_clip` | 12.0 | 早期梯度爆炸时调小 |
| `amp_dtype` | `bfloat16` | 平台 CUDA12.8 支持；老卡改 `float16` |
| `checkpoint_metric` | `val_dice_peri` | 选 `best.pth` 的依据；也可 `val_dice_mean` |
| `model.*` | mednext / base 32 / depth 4 | `arch: resunet` 回退轻量版；`depth: 5` + `max_ch` 换大模型 |
| `global_view.every` | 4 | **每 N 个 step 做一次整脑全局前向**（分类/特殊/嵌入头）。调大省时间 |
| `global_view.size_mm / out` | 192 / 96 | **必须与推理侧一致**（`preprocess.yaml → inference.global_size`），别改 |
| `aux.special_batch` | 2 | 每步目标①②样本数；为 0 则该头不训练 |
| `aux.pair_batch` | 2 | 每步重复影像配对数；为 0 则嵌入头不训练 |
| `loss.*` | 见文件 | 六路权重；某个目标明显弱就调大对应项 |
| `folds` / `val_ratio` | 5 / 0.2 | **只有折训练用**，`full` 模式忽略 |

**显存不足时的调整顺序**：`batch_size` 2→1 → `patch_size` `96³`→`80³` → `val_every` 1→2
→ `num_workers` 4→2 → `global_view.every` 4→8 → 关分割 TTA（`preprocess.yaml` 的 `seg_tta_flips: []`，仅推理侧）。

---

## 10. 训练中怎么读日志

**每个 step（每 `log_every=20` 步一行）** —— 能同时看到六路分项在降：

```text
[g4_full] ep0 step20/1331 loss=1.2345 {'seg': 0.8123, 'cls': 0.3012, 'special': 0.4442, 'embed': 0.0221}
```

**每个 epoch 一行** —— 这是选模指标：

```text
[g4_full] epoch 0 loss=0.9876 dice_core=0.6123 dice_peri=0.7011 thr=[0.4, 0.5] (1832s)
```

| 看见什么 | 含义 |
|---|---|
| `loss` 下降、`dice_*` 上升 | 正常 |
| `special ≈ 0.69` 不动 | 该头没有监督信号（`aux.special_batch=0` 或正样本为空），看 `[loss][告警]` |
| `dice_peri` 长期 ≈ 0 | 掩膜没识别 / 验证集没掩膜 / 输入通道全零（看启动时"没有真输入通道"提示） |
| `[trainer] 断点续训：epoch 3/20` | 复用了上次的 `last.pth`（正常，不是重跑） |

两份日志：

| 文件 | 内容 |
|---|---|
| `logs/train_full.log` | 终端 stdout 的 tee（人看的） |
| `$WORKSPACE/logs/g4_full.jsonl` | 规范要求的 JSONL（`timestamp/epoch/step/phase/loss/lr/mode/data_source/checkpoint`，**epoch 从 1 开始**） |

---

## 11. 断点续训与两个必须知道的陷阱

**续训默认开启**（`src/training/trainer.py:363-374`）：只要 `checkpoints/<tag>/last.pth` 存在，
就加载 `model` + `model_ema` + `optimizer`，并从 `epoch+1` 继续。重跑同一条命令即可。

**陷阱 1 —— 续训时学习率会重爬。**
`OneCycleLR` 每次运行都**重新构造**（代码只 `opt.load_state_dict`，没有 `sched.load_state_dict`），
续训那一次的学习率会从 OneCycle 起点再走一遍 warmup。

> 结论：**别用"跑 5 轮 → 停 → 续跑到 20"的方式**。要么一开始就把 `epochs` 设成最终值一次跑完，
> 要么先用 `epochs: 5` 单独跑一个短模型看效果。

**陷阱 2 —— `val_every > 1` 会写出假的 `best.pth`。**
未验证的 epoch 里 `vm` 是全零字典，`metric = dice_peri = 0.0`；`best` 初值是 `-1.0`，
`0.0 > -1.0` 成立 → 保存一个 **dice=0 的 best.pth**。保持 `val_every: 1` 就不会撞上。

**中断后换数据源**：必须 `python3 scripts/13_build_cache.py --force` 并重跑 `01_probe.sh`，
否则缓存里还是旧数据的预处理结果。

**中途停掉是安全的**：最多丢当前这一轮的进度，上一轮的 `last.pth` 还在，下次重跑自动续。

---

## 12. 产物位置

```text
glioma_track4/
├── checkpoints/<tag>/
│   ├── best.pth     ← 按 checkpoint_metric 选优；**导出提交用这个**
│   └── last.pth     ← 每个 epoch 都写（含 optimizer）；断点续训读这个
└── logs/
    ├── train_full.log / train_fold<N>.log
    ├── g4_full.jsonl                  ← 规范 JSONL
    └── eval_external_ens.log          ← 收尾评估结果
$WORKSPACE/logs/<tag>.jsonl            ← 平台目录存在时 JSONL 落这里
$WORKSPACE/cache/                      ← 预处理缓存（CACHE_DIR）
```

`best.pth` 携带的元信息（推理侧依赖，缺一项就会静默加载出错结构）：
`model` / `model_ema` / `cls_spec` / `arch` / `model_cfg` / `global_size` / `global_size_mm` /
`thresholds` / `epoch` / `best_metric` / `special_trained` / `config`。

---

## 13. 收尾：阈值标定 + 评估 + 导出提交权重

### 13.1 一键收尾（推荐）

训练结束后（**不要在训练中途跑**）：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

SKIP_MOCK=1 FOLDS=full bash scripts/16_finalize.sh

# 只想看"用哪些权重评哪些病例"（不跑推理）
FOLDS=full bash scripts/16_finalize.sh --print-split      # → split=external / external: g4_full
```

五步一条龙：

| 步骤 | 做什么 | 产物 |
|---|---|---|
| 1/5 | 集成阈值标定（**external 口径**，写回权重） | — |
| 2/5 | 全图评估：验证集 Dice / NSD / HD95 | `logs/eval_external_ens.log` |
| 3/5 | 目标一/二 + 重复影像评估 | stdout（`tail -30`） |
| 4/5 | 按规范 §5.2 **导出提交权重** | `{WORKSPACE}/checkpoint/…` |
| 5/5 | Mock Competition + 提交前检查清单（`SKIP_MOCK=1` 可跳过） | — |

- `FOLDS` 可以写折号（`"0 1 2"`）或完整 tag（`full` / `g4_full43_full`）。
- 全量权重**必须**接上验证集走 external 口径；没接上它会明确报错让你先
  `export VAL_ROOT=... && bash scripts/01_probe.sh --val`（不会静默用错口径）。

### 13.2 分步（想知道每一步在干什么时）

```bash
# ① 阈值标定（必须用**同一批权重**：要集成就整批一起标）
python3 scripts/14_calibrate_thresholds.py --split external \
    --ckpt checkpoints/g4_full/best.pth --write

# ② 分割指标（验证集 = 最终口径）
CKPT=checkpoints/g4_full/best.pth bash scripts/04_eval.sh --split external

# ③ 目标一/二 + 重复影像
python3 scripts/15_eval_special_dup.py --split external --ckpt checkpoints/g4_full/best.pth
```

⚠️ **`--ckpt` 一定要给**：自动发现只认 `checkpoints/g4_fold*/`，不给就会拿折权重去评
（数字对不上时先查这里）。

### 13.3 哪些指标算得了（全量 + 验证集口径）

| 指标 | 全量（external） | 说明 |
|---|---|---|
| 分割 Dice / NSD / HD95 | **算得了** | 只统计**有掩膜**的验证集病例；`n_no_mask` 是跳过数 |
| 目标一（假人体）/ 目标二（拼接） | 看验证集 | 验证集自带 `fake/`、`compositing/` 才算得了；没有会明确打印"跳过" |
| 重复影像 | 看验证集 | 需要验证集自己的 `duplicate/` 金标准；没有 → 打印"跳过重复影像指标" |
| 结构化字段 / WHO 分级 | **算不了** | 验证集**没有**字段金标准表；要量就用**折内 OOF**（`04_eval.sh --split fold`） |

**一句话：分割看验证集；字段看折内。**

### 13.4 结果在哪 / 别踩的坑

- 分割（external）：`logs/eval_external_ens.log`
- 目标一/二 + 重复：3/5 段的 stdout
- 规范 JSONL：`logs/g4_full.jsonl`（训练）、评估脚本各自的 `.log`
- **别把"单权重评验证集"的数字当最终口径**：验证集与训练集无交集，**集成**（多 seed / 多折）
  才是最终口径。

---

## 14. 提交前自检与本地闭环

```bash
############ ① 导出 + 复核（全量权重；09 内置 g4_full 兜底，但显式传更稳）############
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/09_export_submission.sh full
bash scripts/09_export_submission.sh --verify

############ ② 契约测试（提交工程，含权重路径与输出头断言）############
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 -m pytest tests/ -q                       # 期望全部 passed

############ ③ 用真实权重跑一次本地推理 ############
export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
export COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint
python3 scripts/local_eval.py \
  --dataset /2026aicompetition/datasets/verification \
  --output  /2026aicompetition/workspace/answer/local-001 \
  --evaluation-id local-001

############ ④ 复现平台协议（期望 5 行 PASS）############
python3 scripts/mock_competition.py \
  --dataset   /2026aicompetition/datasets/verification \
  --workspace /2026aicompetition/workspace --timeout 3600

############ ⑤ 答案目录校验 ############
python3 scripts/validate_output.py \
  --dir /2026aicompetition/workspace/answer/local-001 \
  --expect "$(ls /2026aicompetition/workspace/answer/local-001 | grep -v jsonl | paste -sd, -)"

############ ⑥ 算法工程提交前总检查（期望 15 通过 / 0 失败）############
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/23_pre_submit_check.sh                # 加 --quick 跳过耗时项
```

交付要点：

- `09` 会自动归一化 tag：`full → g4_full`；只给折号 `0` 会补成 `g4_fold0`。
- 单权重时 `goal5_segmentation/` 下是 `core.pt`；多权重（多 seed / 多折）时保留 `<tag>.pt` 供集成。
- 推理侧不设 `GLIOMA_CKPT` 时，会自动读取该目标目录下**全部** `.pt` 做集成。
- `09` 默认用 `copy`（跨文件系统 / 打包上传更稳）。
- ⚠️ 别把 1~2 epoch 的冒烟权重留在 `{WORKSPACE}/checkpoint/` 里 —— `--verify` 只查文件齐不齐，
  查不出"训没训过"。

> 本地（无官方数据）跑 `23_pre_submit_check.sh` 会看到 **15 通过 / 1 失败**，
> 失败项固定是「权重未导出」；容器内做完 §13 后应转绿。

启动推理服务（镜像内由 `start.sh` 拉起）：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
CALLBACK_URL="<平台回调地址>" bash scripts/06_platform_serve.sh
```

---

## 15. 验收清单

**数据**

- [ ] `data/manifest.json` 的 `data_source`/`data_root` 指向**当前**数据根，`n_cases` 合理
- [ ] `data/manifest_val.json` 存在，`mask_role_counts` 非空（有 core/peri）
- [ ] 掩膜与影像**同目录**、以 `_mask.nii.gz` 结尾（`find … -name '*_mask.nii.gz' | wc -l` 非 0）
- [ ] 验证集**没有**字段金标准表是**预期**（不必去找）

**训练**

- [ ] 冒烟通过：`checkpoints/smoke_full/best.pth` 出现
- [ ] 启动日志出现 `[trainer] 全量模式：train=… val=…`
- [ ] `checkpoints/g4_full/best.pth` 与 `last.pth` 都在
- [ ] `logs/train_full.log` 与 `logs/g4_full.jsonl` 在写

**评估 / 交付**

- [ ] `FOLDS=full bash scripts/16_finalize.sh --print-split` 输出 `split=external` 且列出 `g4_full`
- [ ] `logs/eval_external_ens.log` 里有分割指标与 `n_no_mask`（跳过数）
- [ ] `bash scripts/09_export_submission.sh --verify` 六个 goal 全 `ok`
- [ ] `bash scripts/23_pre_submit_check.sh` 通过（本机允许"权重未导出"这类容器内步骤）
- [ ] `{WORKSPACE}/checkpoint/` 里**没有**冒烟权重

---

## 附录 A · 路线 B（六个独立训练的模型，可选兜底）

想让六个模型**各自独立训练**（而不是同源的多任务权重）时用这条。
交付结构同样是六个文件，只是内容各不相同。

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/02_build_dataset.sh            # → data/folds.json（路线 B 依赖它）
python3 scripts/24_verify_eval_split.py     # 期望「0 项 WARN」

cd /2026aicompetition/workspace/dcs/glioma_goals
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation

# ① 数据集自检（CPU，秒级，期望 6/6）
python3 smoke_all_goals.py --datasets-only --data "$DATASET_ROOT" --limit 8

# ② 逐目标训练（各占一张卡；单卡就串行）
for g in goal1_authenticity goal2_stitched goal2_duplicate goal3_tumor goal4_diagnosis goal5_segmentation; do (cd $g && python3 train.py --tag exp1 --fold 0 --data "$DATASET_ROOT"); done

# ③ 导出（缺任一 goal 的权重会非 0 退出）
TAG=exp1 bash scripts/export_to_submission.sh
```

| 项 | 说明 |
|---|---|
| 切分 | 用 `glioma_track4/data/folds.json`（**统一口径，六个目标指标才可比**）；不传 `--fold` 会退回 `val_ratio` 自行划分 |
| 产物 | `runs/<tag>_fold<N>/checkpoints/best.pth`（**每个 epoch 结束才写 best.pth**，没跑完一轮就停会没有任何权重） |
| 全量训练 | 原始代码**不支持**（只有折划分 / `val_ratio`）；要"全部训练集"就用路线 A 的 `03_train.sh full` |
| 每个 epoch 的验证 | 引擎**无条件**在 epoch 末尾遍历整个 val（逐例、batch=1、无日志），所以会有一段很长的静默期，**不是卡死** |
| 打印的 `special` | special 头的 BCE 损失（检查级二分类：假人体/拼接），首值≈0.69 = 未训练头初值，属正常 |
| 断点续训 | ❌ 无。所以**做不到"六个目标一轮一轮交替推进"**（每次调用都从 epoch 0 重来） |

> ⚠️ **两条路线导出到同一个 `checkpoint/<goal>/model.pt`，谁后导出谁覆盖。**
> 要提交就只保留一条路线的导出结果，别来回盖。
> 另外 `CLOUD_DESKTOP_RUNBOOK` §6.4 说"单路权重会被 `_check_heads` 拒绝"，按 §0.2 的代码分析，
> `_check_heads` 只看 `special`/`embed` 两组参数是否存在，机制上不会拒绝；但**务必用 §14 的
> `pytest` + `local_eval` 实测**，别只信文档。

---

## 附录 B · 环境变量与关键路径

建议存成文件，免得每次重设：

```bash
mkdir -p /2026aicompetition/workspace/common
cat > /2026aicompetition/workspace/common/env.sh <<'EOF'
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
export VAL_ROOT=/2026aicompetition/datasets/verification
export GLIOMA_DATASET_ROOT="$DATASET_ROOT"      # 路线 B 用的变量名
export CACHE_DIR=/2026aicompetition/workspace/cache
export WORKSPACE=/2026aicompetition/workspace
export GLIOMA_CHECKPOINT_ROOT="$WORKSPACE/checkpoint"
export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
EOF
source /2026aicompetition/workspace/common/env.sh
```

| 变量 | 作用 | 备注 |
|---|---|---|
| `DATASET_ROOT` | 训练集数据根 | 覆盖 `configs/paths.yaml` 的 `raw.track4`；`.../training` 自动下钻 |
| `VAL_ROOT` | 官方验证集根 | 覆盖 `raw.val`；**填了才算启用 external 口径** |
| `CACHE_DIR` | 预处理缓存 | 放私有存储，容器删除不丢；默认已指向 `$WORKSPACE/cache` |
| `WORKSPACE` | 私有存储根 | 日志自动切到 `$WORKSPACE/logs` |
| `GLIOMA_CHECKPOINT_ROOT` | 提交权重根 | 规范 §5.2 的 `checkpoint/<goal>/` |
| `COMPETITION_PIPELINE_FACTORY` | 推理插件工厂 | **不设 = 跑 Dummy 基线**（看起来"跑通了"其实没加载权重） |
| `GLIOMA_LABELS_DIR` | 标注表目录（可选） | 表就在数据集里时**不用设**；乱设会串表 |
| `GLIOMA_CKPT` | 显式指定推理权重（逗号分隔 = 多模型集成） | 一般不用设：不设时 `common.resolve_checkpoints()` 会自动读取该目标目录下全部 `.pt` 做集成 |
| `PY` / `PYTHON` | 解释器 | `01`~`05` 读 `PY`，`16`/`09`/`23`/`08` 读 `PYTHON`；容器里建议两个都设 |

| 关键路径 | 值 |
|---|---|
| 训练集 | `/2026aicompetition/datasets/training/annotation` |
| 官方验证集 | `/2026aicompetition/datasets/verification` |
| 训练工程 | `/2026aicompetition/workspace/dcs/glioma_track4` |
| 提交工程 | `/2026aicompetition/workspace/dcs/Glioma_recognition-main` |
| 提交权重 | `/2026aicompetition/workspace/checkpoint/<goal>/` |
| 答案 | `/2026aicompetition/workspace/answer/<evaluation_id>/` |
| 预处理缓存 | `/2026aicompetition/workspace/cache` |

| 配置文件 | 作用 |
|---|---|
| `configs/paths.yaml` | 数据根 / 验证集根 / 清单 / 缓存 / 导出路径（env 可覆盖） |
| `configs/train.yaml` | 默认训练配置（100 epoch） |
| `configs/train20.yaml` | **全量训练 20 epoch**（本方案用这份） |
| `configs/_smoke_train.yaml` | 2-epoch 冒烟（小模型小 patch，指标无意义） |
| `configs/preprocess.yaml` | 通道顺序、1mm 公共网格、推理 patch/阈值 |
| `configs/labels.yaml` | 14 个结构化字段定义（决定 `cls_spec`） |

---

## 附录 C · 常见报错对照

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| ① | `ValueError: 数据根 ... 指向数据集父目录，其下是平台阶段目录 [...]` | 填到了 `/2026aicompetition/datasets` | 指到具体阶段（`.../training`）。**预期行为，不是 bug** |
| ② | 探针 / 训练扫到 **0 例** | 数据根高了一层，或影像存储没挂到实例 | 先 `python3 scripts/29_locate_dataset_root.py`；`training/` 下只有 `annotation/` 属正常（自动下钻）；连 `annotation/` 都没有 → 回平台勾选数据集或重建实例 |
| ③ | `series_type_rows = 0`、模态全 `other`、报「无任何可用序列」 | 没读到 `annotation/SeriesType.xlsx` | `python3 scripts/30_inspect_table.py --root $DATASET_ROOT --rows 3` 看表和列名；表在非常规位置才 `export GLIOMA_LABELS_DIR=<目录>` 后重跑 01/02 |
| ④ | 探针报 `mask_role_counts={}` | 掩膜被漏搬 / 命名不符 | `find <根> -name '*_mask.nii.gz' \| head`；掩膜必须与影像同目录、以 `_mask` 结尾 |
| ⑤ | `03_train.sh full` 报「需要官方验证集 / 验证集清单里没有一例带掩膜」 | 没配 `VAL_ROOT` 或没跑 `--val`；或验证集真的没掩膜 | 配 `VAL_ROOT` + `bash scripts/01_probe.sh --val`；确实没掩膜 → 用折内 val：`bash scripts/03_train.sh 0` |
| ⑥ | `folds.json` 的 fold 与当前数据不匹配 → 退回 `val_ratio` | 折划分来自另一个数据集 | 按「先 01 再 02」重建；`data/` 是拷来的先删 `manifest.json`/`folds.json` |
| ⑦ | 评估数字低得离谱 | 用了折权重评验证集 | 显式给 `--ckpt checkpoints/g4_full/best.pth` |
| ⑧ | 明明有验证集，`16_finalize.sh` 却说"未接入"、评估走 OOF | 探针异常被吞掉（典型：GBK 控制台打印 ⚠️ 抛 `UnicodeEncodeError`） | 已内置编码兜底；若再遇到，脚本会把探针原文打出来（`⚠️ 验证集探针没有产出结果`），按原文修 |
| ⑨ | `09_export_submission.sh` 说「未找到 g4_fold*/best.pth」 | 只训了全量 | 传 tag：`bash scripts/09_export_submission.sh full` |
| ⑩ | `local_eval` 只跑 Dummy 基线，像"跑通了" | 没设 `COMPETITION_PIPELINE_FACTORY` | `export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline` |
| ⑪ | 提交后分数极低但权重"齐全" | `checkpoint/` 里是演练（1~2 epoch）权重，`--verify` 查不出来 | 演练一律 `WORKSPACE=/tmp/ws_rehearsal`；提交前用正式权重重新导出 |
| ⑫ | `23_pre_submit_check.sh` 卡在「权重未导出」 | 没跑导出 | `bash scripts/09_export_submission.sh` |
| ⑬ | 训练日志 `special ≈ 0.69` 不动 | 该头没有监督信号 | 看 `[loss][告警]`；正常收敛时它会持续下降 |
| ⑭ | `CUDA out of memory` | patch/batch 太大，或与别的训练共卡 | 按 §9 的调整顺序降；共卡时留足显存 |
| ⑮ | 单卡上设了 `CUDA_VISIBLE_DEVICES=1` → `torch.cuda.is_available()` 为 False | 单卡机器只有 0 号卡 | 留空（默认 0）或显式 `=0` |
| ⑯ | 路线 B 训练：日志停在 `step …/1331`，长时间没有新输出，但进程还活着 | epoch 末的**全量验证**逐例跑、且不打日志 | 看 `nvidia-smi` 利用率与 `glioma_goals/<goal>/cache/` 是否在增长；等待即可。第 2 轮起会快很多 |
| ⑰ | 路线 B：`best.pth` 一直不出现 | `best.pth` **每个 epoch 结束**才写 | 等第 1 轮跑完；中途 kill 不会有任何权重 |

---

## 附录 D · 命令速查

| 步骤 | 命令 |
|---|---|
| 训练集探针 | `bash scripts/01_probe.sh` |
| 验证集探针 | `export VAL_ROOT=… && bash scripts/01_probe.sh --val` |
| （可选）缓存 | `python3 scripts/13_build_cache.py --root "$DATASET_ROOT" --workers 8` |
| 冒烟（2 epoch 全量） | `CONFIG=_smoke_train TAG_PREFIX=smoke bash scripts/03_train.sh full` |
| 全量训练 | `CONFIG=train20 bash scripts/03_train.sh full` |
| 折训练 | `bash scripts/03_train.sh 0` / `bash scripts/03_train.sh all 4` |
| 收尾（一键） | `FOLDS=full bash scripts/16_finalize.sh` |
| 阈值标定 | `python3 scripts/14_calibrate_thresholds.py --split external --ckpt checkpoints/g4_full/best.pth --write` |
| 分割评估 | `CKPT=checkpoints/g4_full/best.pth bash scripts/04_eval.sh --split external` |
| 目标一二/重复 | `python3 scripts/15_eval_special_dup.py --split external --ckpt checkpoints/g4_full/best.pth` |
| 导出权重 | `bash scripts/09_export_submission.sh full` |
| 提交前检查 | `bash scripts/23_pre_submit_check.sh` |
| 划分口径自检 | `FOLDS=full bash scripts/16_finalize.sh --print-split` |

## 一句话小结

```bash
# 全量训练（不做交叉验证，train=全部病例，val=官方验证集）
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/01_probe.sh --val
CONFIG=train20 bash scripts/03_train.sh full
SKIP_MOCK=1 FOLDS=full bash scripts/16_finalize.sh
bash scripts/09_export_submission.sh --verify
bash scripts/23_pre_submit_check.sh

# 折训练（要 OOF / 多折集成时）
bash scripts/03_train.sh 0
```
