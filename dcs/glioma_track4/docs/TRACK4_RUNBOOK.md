# 赛道四 · 算法工程照抄手册（glioma_track4）

> **这份文档怎么用**：从"拿到机器"到"导出可提交权重"，**每一步都给可直接复制粘贴的命令 + 期望输出 + 判据**。
> 命令里的路径**只改一次**（§1 的变量块），后面全部照抄。
>
> 三份文档的分工：
> - `README.md` = **参考手册**（配置项、算法细节、目录结构、指标定义）；
> - `docs/CLOUD_DESKTOP_RUNBOOK.md` = **广场桌面环境与三端闭环**（挂载、镜像、离线依赖、服务拉起、联动演练）；
> - **本文 = 算法工程自己的全流程照抄手册**（探针 → 折划分 → 缓存 → 训练 → 收尾 → 导出 → 提交）。

---

## 0. 三条铁律（先看，能省几小时）

1. **先探针、再建折**：只要换了数据根（本地实验 ↔ 官方数据），必须重跑 `01_probe.sh` 再跑 `02_build_dataset.sh`。
   折文件记着病例来自哪份清单，数据根一变、旧折**静默失效**（训练照跑，但病例/标签对不上）。
2. **提交权重只能用正式折**：任何演练/冒烟/bench 权重都**不许**出现在 `$WS/checkpoint`（提交包会整包收集）。
   演练用 `TAG_PREFIX=smoke` + 私有 `CACHE_DIR`；收尾时 `bash scripts/17_reset_for_official.sh` 清场。
3. **提交前必跑 `23_pre_submit_check.sh`**：任何一项 `FAIL` 都不要提交。

---

## 0.1 全流程一页照抄（最短路径）

```bash
############ ① 变量（只改这一块）############
DCS=/2026aicompetition/workspace/dcs
T4=$DCS/glioma_track4
WS=/2026aicompetition/workspace

# 训练集数据根（有 3_serieslabel.csv / 4_masklabel.csv 的那一层）
DSROOT=/2026aicompetition/datasets/training/annotation
# 官方验证集根（可选但强烈建议；实测只有 SeriesType.xlsx，见 §8）
VALROOT=/2026aicompetition/datasets/verification

export PY=python3
export PYTHON=python3            # 注意：16/09/23/08 读的是 PYTHON，两个都设最稳

export WORKSPACE=$WS
export DATASET_ROOT=$DSROOT
export VAL_ROOT=$VALROOT
export CACHE_DIR=$WS/cache                     # 缓存别放系统盘
export GLIOMA_CHECKPOINT_ROOT=$WS/checkpoint   # 提交包收集权重的根目录
cd $T4

############ ② 环境体检 ############
bash scripts/00_setup_env.sh --mode verify
$PY scripts/99_smoke_test.py                     # 期望： [smoke] PASS ✔

############ ③ 数据根（不确定就定位）############
$PY scripts/29_locate_dataset_root.py            # 末行给出 DATASET_ROOT=...
ls "$DATASET_ROOT" | head -3                     # 应看到 32 位检查号目录

############ ④ 探针（训练集 + 验证集）############
rm -f data/manifest.json data/folds.json         # 换数据必须先删旧清单/旧折
bash scripts/01_probe.sh                         # 训练集清单
bash scripts/01_probe.sh --val                   # 官方验证集清单

############ ⑤ 折划分 + 互验 ############
bash scripts/02_build_dataset.sh
$PY scripts/24_verify_eval_split.py              # 期望： 0 项 WARN

############ ⑥ 缓存（首次 10~30 分钟，之后秒级）############
$PY scripts/13_build_cache.py --workers 8

############ ⑦ 训练 ############
bash scripts/03_train.sh all 4                   # 4 折并行（4 卡）
bash scripts/03_train.sh 4                       # 第 5 折（划分逻辑与第 0 折同源）
tail -f logs/train_fold0.log                     # 盯 loss / dice_peri

############ ⑧ 收尾：标定 → 评估 → 导出 → mock → 自检 ############
FOLDS="0 1 2 3 4" bash scripts/16_finalize.sh

############ ⑨ 提交前检查（独立跑一次，别只看 16 的输出）############
bash scripts/23_pre_submit_check.sh
```

> 时间预算（参考）：环境 10 分钟 → 探针 1~5 分钟 → 折划分 1 分钟 → 缓存 10~30 分钟 → 训练每折 3~8 小时（4 卡并行）→ 收尾 30~60 分钟。

---

## 1. 变量表（改一次，后面全用）

| 变量 | 作用 | 本工程默认 | 什么时候必须改 |
|---|---|---|---|
| `PY` | shell 脚本用的解释器（`01`~`05`） | `python3` | 用 conda 时 `export PY=/opt/conda/envs/glioma4/bin/python` |
| `PYTHON` | **`16`/`09`/`23`/`08` 读的是这个** | `python` | 机器上只有 `python3` 时必设，否则 `python: command not found` |
| `WORKSPACE` | 私有可写存储（权重、缓存、提交包） | 平台注入 | 云桌面本地演练时指向本地目录 |
| `DATASET_ROOT` | 训练集数据根（`01_probe.sh` 实参） | `configs/paths.yaml` 的 `data_root` | 每次换数据都要 export 并重跑探针 |
| `VAL_ROOT` | 官方验证集根（`01_probe.sh --val`） | 空 | 要用验证集时 |
| `CACHE_DIR` | 预处理缓存（体积大，**别放系统盘**） | `$WORKSPACE/cache` | 建议显式 export |
| `GLIOMA_CHECKPOINT_ROOT` | 提交包收集权重的根 | `$WORKSPACE/checkpoint` | 与平台约定不一致时 |
| `CONFIG` | 训练配置文件名（`configs/` 下） | `train` | 换架构/超参 |
| `TAG_PREFIX` | 权重目录前缀：`checkpoints/${TAG_PREFIX}_fold<f>/` | `g4`（→ `g4_fold0`） | **演练必用 `smoke`**（→ `smoke_fold0`，防演练权重混进提交包） |
| `PRETRAINED` | 初始化权重路径 | 空 | 第 5 折 / 二次训练 |
| `SPLIT` | `04_eval.sh` 的评估集选择 | `auto` | 显式指定 `fold` / `external` |
| `CKPT` | 显式指定评估权重 | 自动找 | 对比两个权重时 |
| `FOLDS` | `16_finalize.sh` 处理哪几折 | `0 1 2` | 跑满 5 折时 `FOLDS="0 1 2 3 4"` |
| `SKIP_MOCK` | `16_finalize.sh` 跳过 mock 联动 | `0` | 只想快点看评估结果时 `=1` |
| `VAL_LIMIT` | 外部验证评估的例数上限 | 空（全部） | 冒烟时 `=2` |
| `CUDA_VISIBLE_DEVICES` | 指定可见 GPU | 全部 | 多折并行 / 共用机器 |
| `GLIOMA_LABELS_DIR` | 额外搜索金标准表的目录 | 空 | 表不在数据根附近时 |

**一次性写进文件**（省得每次 export）：

```bash
cat > $T4/.env.local <<'EOF'
export PY=python3
export PYTHON=python3
export WORKSPACE=/2026aicompetition/workspace
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
export VAL_ROOT=/2026aicompetition/datasets/verification
export CACHE_DIR=/2026aicompetition/workspace/cache
export GLIOMA_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint
EOF
source $T4/.env.local          # 每个新终端第一件事
```

---

## 2. 第 0 步：环境与自检

### 2.1 环境

```bash
bash scripts/00_setup_env.sh --mode verify                 # 只体检，不动环境
bash scripts/00_setup_env.sh --mode system                  # 在镜像里装缺失依赖（推荐）
bash scripts/00_setup_env.sh --mode fresh --name glioma4    # 内网/离线时建独立 conda 环境
```

期望：末尾打印依赖自检 OK（缺包会逐条列出）。离线机器：先 `bash scripts/00b_prepare_wheels.sh`，再
`bash scripts/00_setup_env.sh --mode fresh --offline --wheels wheels`。

### 2.2 冒烟（必须 PASS）

```bash
$PY scripts/99_smoke_test.py
```

期望（节选）：

```
[smoke] 链路：造数据 → 单步训练 → 导出 → mock 提交 → 答案格式校验
[smoke] PASS ✔ 全链路与答案格式合规
```

`FAIL` 时看它打印的**第一个**失败点：通常是依赖缺失，或 `PY/PYTHON` 指错解释器。

### 2.3 代码级自检（改动过代码才需要）

```bash
$PY scripts/21_verify_fixes.py                 # 历史修复回归，期望 16/16 项通过
$PY scripts/22_fault_injection.py              # 故障注入：坏数据要"报错"，不能"跑出垃圾"
$PY scripts/26_audit_plugin_completeness.py    # 插件/协议完整性，期望 0 项 FAIL
$PY scripts/20_bridge_selftest.py --n 3        # 工程师侧桥接自测（需要工程师侧仓库）
```

---

## 3. 第 1 步：定位数据根（不知道 `DATASET_ROOT` 填什么时）

```bash
$PY scripts/29_locate_dataset_root.py
```

期望（末行给出可直接复制的变量）：

```
[29] 候选数据根（含 3_serieslabel.csv / 4_masklabel.csv 或 <检查号>/<SeriesUid>/ 结构）：
[29]   /2026aicompetition/datasets/training/annotation   病例 1234
[29] DATASET_ROOT=/2026aicompetition/datasets/training/annotation
```

| 检查 | 期望 |
|---|---|
| `ls "$DATASET_ROOT"` | 看到 32 位十六进制检查号目录（若有 `original/` 之类中间层，探针会自动下钻） |
| 表格 | 该目录（或父/祖父层）能找到 `3_serieslabel.csv` / `4_masklabel.csv` |

**坑**：别把数据根定太深（例如定到 `…/original` 里的病例目录），探针只在"数据根 + 上两级"找表。定位脚本给哪一层就用哪一层。

---

## 4. 第 2 步：探针（01）

探针是**唯一**的数据体检，它决定后面标签对不对。

```bash
rm -f data/manifest.json data/folds.json         # 换数据先删旧清单/旧折
bash scripts/01_probe.sh                         # 训练集 -> data/manifest.json
bash scripts/01_probe.sh --val                   # 官方验证集 -> data/manifest_val.json
```

期望输出（训练集，节选）：

```
[probe] 扫描数据根：/2026aicompetition/datasets/training/annotation
[probe] 训练清单 -> data/manifest.json   病例 1234（含掩膜 1234）
[probe] 模态：T1c 1234 / T1 1234 / T2 1234 / FLAIR 1234
[probe] 掩膜角色：core 1234 / peri 1234
[probe] 结构化字段：WHO_Grade 1234 / TumorProbability 1234 / location 1234 …
```

**验收**（打印 `data/manifest.json` 的报告段）：

```bash
$PY - <<'PY'
import json
r = json.load(open("data/manifest.json"))["report"]
for k in ("n_cases","series_type_rows","modality_counts","mask_role_counts",
          "label_field_counts","no_labels","missing_t1c","missing_flair","special"):
    print(f"{k:20s} {r.get(k)}")
print("labels_hint:", (r.get("labels_hint") or "")[:120])
PY
```

| 字段 | 判据 | 不达标怎么办 |
|---|---|---|
| `n_cases` | 与官方病例数一致 | 差很多 → 数据根不对（回 §3） |
| `modality_counts` | 4 个模态都接近满员 | 缺 T1c/FLAIR 的病例看 `missing_t1c/missing_flair`，真缺就接受 |
| `mask_role_counts` | `core`/`peri` 都≈全量 | 只有 core 没有 peri → 查 `4_masklabel.csv` 是否被读到 |
| `label_field_counts` | 训练集应**非空** | 全空 → 表没找到：`export GLIOMA_LABELS_DIR=<含表目录>` 或 `ln -s` 到 `<工程>/labels`，重跑 |
| `no_labels` | 训练集应**很短** | 长列表说明大量病例没字段标签，分类头学不到东西，先解决表 |

**验证集（`--val`）的期望与之不同，这是正常的**：

```
[probe] ℹ️ 结构化字段金标准为空（label_field_counts={}）：这是**验证集**：官方验证集实测只有
        SeriesType.xlsx、**没有**字段金标准表 → 本级 labels 全空属**预期**，不必去找表…
[probe] 验证清单 -> data/manifest_val.json   病例 200
```

看到 `ℹ️`（不是 `⚠️`）+ `no_labels` 列出全部病例 = **正常**，不是配置错误，详见 §8。

> 想单独看某张表解析成什么样：`$PY scripts/30_inspect_table.py --rows 5 <表文件>`。

---

## 5. 第 3 步：折划分（02）与互验（24）

```bash
bash scripts/02_build_dataset.sh          # 5 折，<工程>/data/folds.json
$PY scripts/24_verify_eval_split.py       # 折划分互验
```

期望：

```
[02] 清单来源：data/manifest.json   数据根：…   病例 1234（含标注 1234）
[02] folds: 0: train 987 / val 247 ; 1: train 987 / val 247 ; …
[02] 折文件 -> /…/glioma_track4/data/folds.json
```

```
[24] 折划分互验：N 项通过，0 项 WARN，0 项 FAIL
```

| 检查 | 期望 | 说明 |
|---|---|---|
| 每折 val 例数 | 大致相等（±1 例） | 差得多说明有病例缺模态被剔，报告里会列出 |
| `24` 的 WARN | **0 项** | "重复影像跨折"=泄漏；"病例从未进入任何折 val"=拿不到 OOF |
| 第 5 折 | `fold 4` 的 train 是**全量** | 划分逻辑与第 0 折同源 |

---

## 6. 第 4 步：缓存（13）

```bash
$PY scripts/13_build_cache.py --workers 8
```

期望：逐例进度，末尾给出缓存目录与体积；**第二次运行应当秒级完成**（全命中）。

| 现象 | 处理 |
|---|---|
| 磁盘不足 | `export CACHE_DIR=$WS/cache`（别用系统盘/`/tmp`），重跑 |
| 慢（每例 >3s） | 提高 `--workers`（≤ CPU 核数）；DICOM 逐层读本身慢，正常 |
| 换数据想重来 | 删掉 `$CACHE_DIR` 下对应数据源的子目录 |

---

## 7. 第 5 步：训练（03）

### 7.1 四种跑法

| 目的 | 命令 | 说明 |
|---|---|---|
| 单折（先验通链路） | `bash scripts/03_train.sh 0` | 单卡 1 折 |
| 4 折并行 | `bash scripts/03_train.sh all 4` | 按可见 GPU 分配（`CUDA_VISIBLE_DEVICES` 控制） |
| 第 5 折 | `bash scripts/03_train.sh 4` | 用全部病例训练的那一折 |
| 全量训练（用官方验证集早停） | `bash scripts/03_train.sh full` | **需要验证集带掩膜**，见 §8 |

### 7.2 日志怎么读

```bash
tail -f logs/train_fold0.log
```

```
[trainer] 数据源标识: train=local/training/annotation val=local/training/annotation
[trainer] tag=g4_fold0 fold=0 train=987 val=247
[g4_fold0] ep0 step40/247 loss=1.2381 {'dice': 0.004, 'cls': 0.693}
[g4_fold0] epoch 0 loss=1.2013 dice_core=0.0000 dice_peri=0.0000 thr=[0.4, 0.5] (812s)
[g4_fold0] epoch 40 loss=0.2103 dice_core=0.8712 dice_peri=0.8120 thr=[0.45, 0.55] (798s)
[trainer] done. best=0.8120 thr=[0.45, 0.55] -> checkpoints/g4_fold0/best.pth
```

判据：
- 前若干 epoch `dice_*` **接近 0 是正常的**（从零训练）；若到 20 epoch 仍为 0 → 回看 §4 的 `mask_role_counts`；
- 选模指标是 `dice_peri`，应稳定上升；`thr` 随之变化属正常；
- 每 epoch 末尾括号是该 epoch **耗时秒数**，用它估总时长；
- 结构化日志另有 `logs/*.jsonl`（可视化/事后分析用）。

### 7.3 断点续训 / 换权重 / 换配置

```bash
bash scripts/03_train.sh 0                  # 同命令再跑一次 = 自动读 last.pth 续训
bash scripts/03_train.sh 0 --no-resume      # 想从头再来
CONFIG=train_xxx bash scripts/03_train.sh 0 # 换配置（新建 configs/train_xxx.yaml，别乱改 train.yaml）
PRETRAINED=checkpoints/g4_fold0/best.pth bash scripts/03_train.sh 4   # 用已有权重初始化
```

**演练必须隔离**（铁律 2）：

```bash
TAG_PREFIX=smoke CACHE_DIR=$WS/cache_smoke bash scripts/03_train.sh 0
# 结果落在 checkpoints/smoke_fold0/，不会进提交包
```

### 7.4 显存与耗时

| 情况 | 期望/处理 |
|---|---|
| 24G 卡、`batch=2`、`96³` patch | 单折可跑；4 卡并行 4 折 |
| OOM | 降 `batch`（`configs/model.yaml`）或 `CUDA_VISIBLE_DEVICES=0,1` 只跑 2 折 |
| 单折时长 | 3~8 小时（取决于折内例数与早停前 epoch 数） |

---

## 8. 验证集没有金标准，能不能用它做全量训练的验证？有影响吗？

**结论：能，前提是验证集病例带掩膜（分割金标准）。字段金标准缺失没有任何影响。**

### 8.1 为什么可以

`--fold full` 的验证集只干一件事：每 `val_every` 个 epoch 算一次 Dice（`checkpoint_metric: val_dice_peri`）来选 `best.pth`。
这是**纯分割指标**，只需要掩膜：

| 验证集有什么 | 对全量训练（`--fold full`）的作用 |
|---|---|
| 掩膜（`masks`） | **必须** —— 选模/早停的唯一依据 |
| 字段金标准（`labels`） | **完全不用** —— `validate()` 只用 `batch["target"]`，`labels` 在 val 里不参与任何计算 |
| 重复影像金标准 | 不用（那是评估口径，不是训练信号） |

代码里已钉死这点：`external_val_cases()` **只保留 `masks` 非空的病例**（`src/utils/config.py`）。
此前是 `masks or labels` —— 那样"只有标签、没有掩膜"的病例会混进 val，而它们让 `make_targets` 产出**全零 target**，
Dice 在"预测也为空"时按 `den == 0` 记成 **1.0 的假满分**（`src/training/trainer.py:285`），**把 best 直接选歪**。

### 8.2 缺字段金标准的影响清单

| 方面 | 有无影响 | 说明 |
|---|---|---|
| 训练监督（目标三/四：分级、位置、形态、信号） | **无** | 监督信号来自**训练集**的 `脑胶质瘤标注结果-训练集.xlsx`，与验证集无关 |
| 全量训练的早停/选模 | **无** | 只看 Dice |
| 官方评分口径 | **无** | 官方评估 = Dice/NSD/HD95 + 重复影像，**不评字段** |
| "在验证集上报告字段指标" | 有 | 本来就做不了，且官方不要求；算不出属预期 |
| 重复影像（`duplicate_w*`） | 有 | 验证集没给重复对金标准 → 报告里没有 `duplicate_w*` 键；`15_eval_special_dup.py --split external` 会打印"评估集内无正样本，跳过 AUC"。这项只能在折内 OOF / 本地数据看趋势 |

### 8.3 一条命令判断"我的验证集能不能用于 full"

```bash
# 先确保跑过： bash scripts/01_probe.sh --val
$PY - <<'PY'
import json
d = json.load(open("data/manifest_val.json", encoding="utf-8"))
cs = d.get("cases") or []
with_mask = [c for c in cs if c.get("masks")]
with_lab  = [c for c in cs if c.get("labels")]
dup = (d.get("special") or {}).get("gold_pairs") or []
print(f"验证集病例        : {len(cs)}")
print(f"带掩膜的病例      : {len(with_mask)}   ← full 模式早停只需要这个，>0 就能用")
print(f"带字段标签的病例  : {len(with_lab)}    ← 官方评估不评字段，0 也正常")
print(f"重复影像金标准对  : {len(dup)}         ← 0 表示 duplicate_w* 不会有值")
print(f"探针 no_labels    : {len((d.get('report') or {}).get('no_labels') or [])}")
PY
```

- `带掩膜的病例 > 0` → 直接 `bash scripts/03_train.sh full`，会打印：
  `[trainer] 全量模式：train=N（清单全部病例） val=M（官方验证集 …；无掩膜的例已剔除——选模只用 Dice）`
- `带掩膜的病例 == 0` → `full` **明确报错**（不会再让你去重跑探针）：

```
[trainer] 全量训练（--fold full）需要官方验证集：
  · 验证集清单里**没有一例带掩膜**（Dice 算不出来，选不了 best）（清单：…/data/manifest_val.json）
  · 首次生成：export VAL_ROOT=<验证集根> && bash scripts/01_probe.sh --val
  · 用不了验证集时请用折内 val：bash scripts/03_train.sh 0
```

  此时用折内 val（`bash scripts/03_train.sh 0..4`）即可。

### 8.4 建议

- 验证集例数少（几十例）时 `best.pth` 抖动大 → 在 `configs/train.yaml` **固定 `epochs`**，别纯靠早停；
- 官方验证集与训练集**无交集**，是最干净的无泄漏早停集，能用就用；
- 平台若后续单独发布验证集标注表：放进验证集目录任意位置（或 `export GLIOMA_LABELS_DIR=<含表目录>`），
  重跑 `bash scripts/01_probe.sh --val` 即自动接上。

---

## 9. 第 6 步：收尾（16）

```bash
FOLDS="0 1 2 3 4" bash scripts/16_finalize.sh                            # 完整收尾
SKIP_MOCK=1 VAL_LIMIT=2 FOLDS="0" bash scripts/16_finalize.sh             # 冒烟版（几分钟）
FOLDS="0 1 2 3 4" bash scripts/16_finalize.sh --print-split               # 只看"用哪些权重评哪些病例"
```

**口径**（脚本自己判定，不看你口头怎么说）：

| 条件 | 口径 | 选模/评估方式 |
|---|---|---|
| 有可用 `data/manifest_val.json` | `external` | **全折集成**评官方验证集，**不做留一**（验证集与训练集无交集） |
| 没有 | `oof` | **留一折集成**（评第 f 折时只用其余折的模型，避免该折模型见过本折 val） |

启动横幅会明确告诉你在哪套口径：

```
[finalize] 官方验证集：200 例 …/data/manifest_val.json
[finalize] → 采用 external 口径（全折集成，不做留一）
```

或

```
[finalize] 未接入官方验证集（无 data/manifest_val.json）→ 回退折内 val（OOF/留一）
[finalize]   想接入：export VAL_ROOT=<验证集目录> && bash scripts/01_probe.sh --val
```

5 个步骤与期望：

| 步骤 | 内容 | 期望输出 |
|---|---|---|
| 1/5 | 集成阈值标定（写回所有折） | `14_calibrate_thresholds.py` 打印每折 `thr_core/thr_peri`，落 `checkpoints/g4_fold*/calib.json` |
| 2/5 | 全图评估 | external：`logs/eval_external_ens.log`；oof：`logs/eval_ens_fold*.log`，尾部有 `dice_core/dice_peri/dice_mean` |
| 3/5 | 目标一/二 + 重复影像 | external 用验证集自带 special；否则 OOF。**没有重复金标准时会打印"评估集内无正样本，跳过 AUC"** |
| 4/5 | 导出提交权重 | 见 §11（`goal5_segmentation/` 下的 `.pt`） |
| 5/5 | Mock Competition + 提交前检查 | `PASS Callback` + `汇总：N 通过 / 0 失败` |

收尾末尾会打印提交要点（照它做）：

```
[finalize] 完成。提交要点：
  · 最终数字看哪份日志：external 口径 → logs/eval_external_ens.log
  · 多折集成无需手动指定权重：resolve_checkpoints() 自动读
    /2026aicompetition/workspace/checkpoint/goal5_segmentation/ 下的全部 .pt
  · 测评容器启动命令：CALLBACK_URL="<平台回调地址>" bash scripts/06_platform_serve.sh
  · 若检查清单有失败项，**不要提交**。
```

---

## 10. 第 7 步：单独评估（想复现某个数字时）

```bash
bash scripts/04_eval.sh                                  # auto：有验证集就评验证集
bash scripts/04_eval.sh --split external                 # 强制官方验证集（全折集成）
bash scripts/04_eval.sh --split fold 1                   # 强制折内 val（留一集成）
bash scripts/04_eval.sh 1 30                             # fold=1，只看 30 例（快）
CKPT="a.pth,b.pth" bash scripts/04_eval.sh --split external   # 指定集成权重

$PY scripts/14_calibrate_thresholds.py --split external --limit 50 --write   # 标定阈值并写回
$PY scripts/15_eval_special_dup.py --split external      # 目标一/二 + 重复影像
$PY scripts/15_eval_special_dup.py --split oof           # 同上，但用 OOF（无泄漏）

$PY scripts/12_bench.py                                  # 推理延迟/显存基准（报"能不能过时限"）
$PY scripts/11_dicom_selftest.py                         # DICOM 解码自测
```

判据：
- `dice_mean` 与 `16_finalize.sh` 的 2/5 一致（同一口径、同一权重）；
- 阈值改动后必须重跑评估（`14` 写回 → `04` 复评），否则数字与提交物不匹配；
- `12_bench.py` 的单例延迟 × 病例数要**留 2 倍余量**（测评机器可能更慢）。

---

## 11. 第 8 步：导出可提交权重（09 / 07）

```bash
bash scripts/09_export_submission.sh                       # 默认导出 g4_fold0,1,2
bash scripts/09_export_submission.sh g4_fold0,g4_fold1     # 指定 tag
bash scripts/09_export_submission.sh --verify              # 只校验现有导出
```

期望：

```
[09] 导出：checkpoints/g4_fold0/best.pth -> /…/workspace/checkpoint/goal5_segmentation/…pt
[09] 已完成 3 个权重导入；目录：/…/workspace/checkpoint/goal5_segmentation
[09] 复核：目标一/目标二权重目录结构符合规范 §5.2 ✔
```

红线（**检查清单会替你查，但你自己别踩**）：

| 红线 | 后果 |
|---|---|
| 导出演练权重（`smoke*/bench*`） | 提交的权重是没训练的 → 分数崩，且**无法挽回**（提交后不可替换） |
| 只导出 1 折 | 少了集成收益，通常掉 1~3 个 Dice 点 |
| 忘了 `07_export_weights.sh` 持久化 | 容器重启后私有存储里的权重丢了 |

```bash
bash scripts/07_export_weights.sh      # 把权重/标定持久化到私有存储（防容器重置）
```

---

## 12. 第 9 步：提交前检查（23）

```bash
bash scripts/23_pre_submit_check.sh          # 全量
bash scripts/23_pre_submit_check.sh --quick  # 跳过耗时项（自检/桥接/压测/mock）
```

检查项与期望：

| 项 | 期望 |
|---|---|
| `.py` 语法 | `全部 .py 语法通过` |
| 关键模块导入 | `关键模块导入通过` |
| 团队测试（有工程师侧仓库时） | `团队测试通过（含 fail-fast 契约断言）` |
| P0 修复回归 | `P0 修复 16/16 断言通过` |
| 端到端自检 | `端到端自检 PASS` |
| 桥接集成 | `桥接集成 PASS` |
| 故障注入 | `故障注入 22/22 均有合规答案` |
| Mock Competition | `Mock Competition 协议闭环通过` |
| `.gitignore` / 入库范围 | 源码入库、`data/*.json` 产物排除、**`data/modality_model.json` 必须随库分发**、`data/folds.json` 不得入库 |
| 硬编码本地路径 | `运行时代码无硬编码本地路径` |
| 评估划分逻辑 | `评估划分逻辑自检通过（含 external 全折集成与 OOF 语义端到端验证）` |
| 权重已导出 | `权重已导出（goal5_segmentation 下 N 个 .pt）` |

末行汇总：

```
汇总：24 通过 / 0 失败 / 0 跳过
```

**`失败 > 0` 就不要提交。** 常见失败与处理：

| 失败项 | 处理 |
|---|---|
| `P0 断言未全通过` | `$PY scripts/21_verify_fixes.py` 看是哪条，回滚最近改动 |
| `data/modality_model.json 被 .gitignore 吞掉` | 改 `.gitignore`，`git add -f data/modality_model.json`（评测期模态判定全靠它） |
| `权重未导出` | 跑 `bash scripts/09_export_submission.sh` |
| `存在硬编码本地路径` | 把 `/mnt/...`、`/tmp/...` 换成 `configs/paths.yaml` / 环境变量 |

---

## 13. 第 10 步：服务与协议闭环（05 / 06 / 08）

### 13.1 本地起服务自测（不用真实数据）

```bash
CKPT=checkpoints/g4_fold0/best.pth PORT=8000 bash scripts/05_serve.sh
```

它自测 `/health` + 规范 `/call`（嵌套体）与扁平兼容写法。期望打印健康检查与一次成功回调。

### 13.2 平台"测评容器"启动命令（填到实例里）

```bash
CALLBACK_URL="http://<平台地址>/api/competition/inference/callback/" \
  bash /2026aicompetition/workspace/glioma_track4/scripts/06_platform_serve.sh
```

平台硬性要求（本工程已满足）：

| 要求 | 本工程行为 |
|---|---|
| 8000 端口 + `/health` 常活（启动探针 5s/次、最长 1800s） | 常驻 |
| 启动命令是**长期运行的前台进程** | `06` 前台运行 |
| `/call` 必须 5s 内回 200 | 后台异步：立即 200，推理完再回调 |

⚠️ **`CALLBACK_URL` 必须通过环境变量传入**：不传则推理完成后无法通知平台闭环，**该次测评不计分**。地址在容器实例页面顶部。

### 13.3 Mock Competition（提交前必过）

```bash
bash scripts/08_mock_competition.sh                          # 默认 3 折集成
bash scripts/08_mock_competition.sh g4_fold0,g4_fold1        # 指定折
GLIOMA_CKPT=/abs/a.pth,/abs/b.pth bash scripts/08_mock_competition.sh   # 指定权重
```

期望 5 项全 PASS：

```
PASS Health                     ← /health 常活
PASS Call response time         ← /call 5s 内 200
PASS Background / Callback      ← 后台推理 + 完成回调
PASS Output and NIfTI validation← prediction.json + 掩膜结构与几何合规
PASS 汇总
```

> Mock 用的是**合成小数据集**，验证的是**协议闭环**，不是精度 —— 精度看 §9 的评估日志。

---

## 14. 切换到正式数据 / 重训（10 / 17 / 18 / 19 / 31 / 32 / 33）

### 14.1 本地实验数据（BraTS）只用于验证链路

```bash
$PY scripts/10_brats_to_track4.py ...    # 把 BraTS 整理成赛道四布局
```

**合规红线**：正式提交的权重必须只用大赛提供的数据训练；BraTS 产物只能用来验证工程链路（探针/掩码归位/1mm 网格/答案格式）与算法上限，**不可**用来产出提交权重。

### 14.2 切官方数据前的清场（必做）

```bash
bash scripts/17_reset_for_official.sh
```

把本地验证产物（清单/折/缓存/权重/日志）**移动**到 `_local_validation_backup/`（**不删除**，可随时恢复），避免：
- 与官方数据产物混用；
- `assert_data_source` 闸门拒绝"清单数据源与本机数据源不同类"的训练；
- 提交物里的 `logs/*.jsonl` 混进非官方数据的训练记录。

清场后**必须重跑**：`01_probe.sh` → `02_build_dataset.sh` → `13_build_cache.py`（铁律 1）。

### 14.3 折划分有缺陷时：从零重训（别混搭）

```bash
bash scripts/18_retrain_new_folds.sh
```

为什么必须**从头**重训：旧权重的训练集基于**旧划分**，新旧划分随机独立、val 会重叠约 80%。拿旧权重按新划分评估 = 模型评到自己训练过的病例 = **严重泄漏**。所以旧权重只能配旧划分，二者不可混搭。

它做六件事（幂等、不删东西）：停训练 → 归档旧产物 → 重探针 → 生成新折并当场校验 → 重建缓存 → 4 折并行训练（第 5 折随后 `bash scripts/03_train.sh 4`）。

### 14.4 无人值守重训（先保住一个可提交版本）

```bash
bash scripts/19_safe_retrain_pipeline.sh
```

重训是 14~19 小时的长任务，且归档旧产物后旧权重**不可复用**。本流水线**先按规范导出当前折作为"保险"**（`$BACKUP_ROOT/checkpoint/<goal>/`，校验权重数与目录齐全），再执行清场+重训 —— 这样重训无论成败，手上始终有一个**完整可提交**的版本。

### 14.5 模态兜底与标签修补

```bash
$PY scripts/31_train_modality_model.py --root "$DATASET_ROOT"   # 训练 data/modality_model.json
bash scripts/33_fix_abnormal_labels.sh                           # 修 1_abnormal 的 Label（来源）列并重跑探针
$PY scripts/32_apply_abnormal_patch.py ...                       # 核验并合并人工补丁（按 Label 拼路径核对命中率）
```

- `31`：**评测集的序列目录名是 DICOM UID**，拿不到 `SeriesType.xlsx`，靠关键词一个模态都挑不出来 → 用体素统计 + 逻辑回归训练一个几百参数的兜底模型。`data/modality_model.json` **必须随库分发**（`23` 会专门查它有没有被 `.gitignore` 吞掉）。
- `33`/`32`：`1_abnormal.xlsx` 的 `Label` **不是"真/假"，而是"来源"**（`true`/`fake`/`compositing`/`duplicate`），官方就是用它拼路径的。改前先用 `33` 核验"改前 → 改后"的磁盘命中率，命中率上升才落地（自动备份原表）。

---

## 15. 自检矩阵（一条命令一个"体检项"）

| 命令 | 覆盖 | 期望 |
|---|---|---|
| `$PY scripts/99_smoke_test.py` | 造数据 → 训练 → 导出 → mock → 答案格式 | `PASS ✔` |
| `$PY scripts/21_verify_fixes.py` | 历史修复回归 | `16/16` |
| `$PY scripts/22_fault_injection.py` | 故障注入（坏数据要报错，不能出垃圾） | `22/22 均有合规答案` |
| `$PY scripts/24_verify_eval_split.py` | 折划分 + 评估口径（含 external/OOF 语义） | `0 项 WARN` |
| `$PY scripts/25_verify_tasks_integration.py` | 任务集成（探针/清单/协议/认证） | `161/161` |
| `$PY scripts/26_audit_plugin_completeness.py` | 插件/协议完整性 | `0 项 FAIL` |
| `$PY scripts/20_bridge_selftest.py --n 3` | 与工程师侧桥接 | `[bridge] PASS` |
| `$PY scripts/12_bench.py` | 延迟/显存基准 | 单例延迟 × 病例数留 2 倍余量 |
| `$PY scripts/11_dicom_selftest.py` | DICOM 解码 | 全通过 |

---

## 16. 报错对照表（照这个修，别乱猜）

| 现象 | 原因 | 处理 |
|---|---|---|
| `python: command not found` | `16/09/23/08` 读的是 `PYTHON` | `export PYTHON=python3`（同时设 `export PY=python3`） |
| `找不到 python3/python：请 export PY=` | 解释器不在 PATH | `export PY=/opt/conda/envs/xxx/bin/python` |
| `[trainer] 全量训练（--fold full）需要官方验证集` | 没有 `manifest_val.json`，或**一例带掩膜的都没有** | §8：跑 `01_probe.sh --val`，或改折内 val |
| `清单数据源与本机数据源不同类` | 清单与当前数据根不匹配 | 重跑 `01_probe.sh`（铁律 1）；换官方数据先 `17_reset_for_official.sh` |
| 训练照跑但指标离谱 / `train` 与 `val` 重叠 | 旧 `folds.json` 配了新清单 | `rm -f data/folds.json && bash scripts/02_build_dataset.sh` |
| 训练集探针 `label_field_counts` 全空 | 没找到字段金标准表 | `export GLIOMA_LABELS_DIR=<含表目录>` 或 `ln -s` 到 `<工程>/labels`，重跑探针 |
| 验证集探针 `no_labels` 列出全部病例、`ℹ️` 提示 | **正常**：官方验证集没有字段金标准表 | 不用处理，见 §8 |
| 报告里没有 `duplicate_w*` 键 / `15` 说"无正样本，跳过 AUC" | 没有重复影像金标准 | 正常；用 OOF 或本地数据看趋势 |
| `24` 报"重复影像跨折" | 同一病人的重复序列落在不同折 | 修折划分（`02` 支持按病人分组），重训（§14.3） |
| OOM | 显存不够 | 降 `batch`（`configs/model.yaml`）或 `CUDA_VISIBLE_DEVICES=0,1` 少跑一折 |
| `23` 报 `data/modality_model.json 被 .gitignore 吞掉` | `.gitignore` 规则过宽 | `git add -f data/modality_model.json`（评测期模态判定全靠它） |
| `23` 报 `权重未导出` | 没跑导出 | `bash scripts/09_export_submission.sh` |
| `23` 报 `存在硬编码本地路径` | 代码里写死了 `/mnt/...`、`/tmp/...` | 改成 `configs/paths.yaml` / 环境变量 |
| mock 不出 `PASS Callback` | 没传 `CALLBACK_URL` 或服务没起 | §13.2 的启动命令带上 `CALLBACK_URL` |
| 评测提交后不计分 | 服务没按协议跑 / 权重是演练权重 | §13.2、§11 红线 |

---

## 17. 最终验收清单（打印贴墙）

```
[ ] 环境：scripts/00_setup_env.sh --mode verify 通过；99_smoke_test.py PASS
[ ] 数据根：29_locate_dataset_root.py 定位；01_probe.sh（训练集）n_cases 与官方一致
[ ] 字段监督：训练集 label_field_counts 非空（分类头有监督信号）
[ ] 验证集：01_probe.sh --val 生成 manifest_val.json（labels 全空属预期，§8）
[ ] 折划分：02 完成且 24 报 0 WARN
[ ] 缓存：13 跑完且二次运行秒级命中
[ ] 训练：折内 dice_peri 收敛；checkpoints/g4_fold*/best.pth 齐全（含第 5 折）
[ ] 收尾：16_finalize.sh 跑完；logs/eval_external_ens.log（或 eval_ens_fold*.log）有最终数字
[ ] 标定：14 已写回各折阈值，且 2/5 的评估用的是新阈值
[ ] 闭环：08_mock_competition.sh 五项 PASS
[ ] 导出：09_export_submission.sh 完成；GLIOMA_CHECKPOINT_ROOT/goal5_segmentation/ 下有权重
[ ] 持久化：07_export_weights.sh 已跑（容器重启不丢）
[ ] 清场：17_reset_for_official.sh 已跑；提交物 logs/*.jsonl 全部来自官方数据训练
[ ] 提交前：23_pre_submit_check.sh 汇总"0 失败"
[ ] 服务：06 启动命令带 CALLBACK_URL；/health 常活
```

---

## 附：文档索引

| 想知道什么 | 看哪里 |
|---|---|
| 命令怎么敲、期望什么输出、报错怎么办 | **本文（TRACK4_RUNBOOK.md）** |
| 配置项含义、算法细节、目录结构、指标定义 | `README.md` |
| 广场桌面/云桌面环境、镜像、离线依赖、三端联动 | `docs/CLOUD_DESKTOP_RUNBOOK.md` |
| 整体架构与三工程协作 | `docs/MASTER_GUIDE.md` |
| 赛事规范原文 | 平台《赛事开发规范》 |




