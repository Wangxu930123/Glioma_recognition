# 赛道四 · 云平台使用文档（上传 → 全量训练 → 提交）

> **适用场景**：单卡（或单容器）、**不做 N 折交叉验证**、用**全部训练集训练 20 epoch**、
> 用**官方验证集**做验证，最后导出提交权重。
>
> **主路线 = 路线 A**（`glioma_track4` 一体化多任务训练）；**兜底 = 路线 B**（`glioma_goals` 六目标）。
> 两条路线二选一即可，权重都落到同一个 `checkpoint/` 契约目录。
>
> 本文是**操作手册**。权威口径见：`glioma_track4/docs/MASTER_GUIDE.md`（三工程总说明）、
> `glioma_track4/docs/CLOUD_DESKTOP_RUNBOOK.md`（克隆/搬迁/排错）、
> `glioma_track4/docs/TRACK4_RUNBOOK.md`（算法工程逐步骤）。

---

## 0. 五分钟速览（照抄版）

```bash
############ 变量：按你机器上的实际路径改一次 ############
DCS=/2026aicompetition/workspace/dcs
DSROOT=/2026aicompetition/datasets/training/annotation    # 训练集病例层
VALROOT=/2026aicompetition/datasets/verification          # 官方验证集根
WS=/2026aicompetition/workspace

export DATASET_ROOT="$DSROOT"
export VAL_ROOT="$VALROOT"
export CACHE_DIR="$WS/cache"
export WORKSPACE="$WS"
export GLIOMA_DATASET_ROOT="$DSROOT"     # glioma_goals 用的变量名，一起设免得踩

############ ① 依赖 + 合成自检（不需要真实数据）############
cd "$DCS/glioma_track4"
bash scripts/00_setup_env.sh --mode system

############ ② 数据根定位（不确定才跑）############
python3 scripts/29_locate_dataset_root.py

############ ③ 探针 + 折划分 ############
bash scripts/01_probe.sh                 # → data/manifest.json
bash scripts/02_build_dataset.sh         # → data/folds.json（兜底路线 B 要用）
python3 scripts/24_verify_eval_split.py  # 期望「0 项 WARN」

############ ④ 官方验证集清单（全量训练的 val）############
bash scripts/01_probe.sh --val           # → data/manifest_val.json

############ ⑤ 冒烟：先证明链路通（几分钟）############
CONFIG=_smoke_train TAG_PREFIX=smoke bash scripts/03_train.sh full
#   → checkpoints/smoke_full/best.pth 出现 = 通

############ ⑥ 正式全量训练 20 epoch ############
CONFIG=train20 nohup bash scripts/03_train.sh full > logs/full20_launch.log 2>&1 &
tail -f logs/train_full.log

############ ⑦ 收尾：评估 + 导出提交权重 ############
SKIP_MOCK=1 FOLDS=full bash scripts/16_finalize.sh
bash scripts/09_export_submission.sh --verify

############ ⑧ 提交前自检 ############
bash scripts/23_pre_submit_check.sh
```

---

## 1. 上传后的目录落位

三个工程必须**并排**放在同一个父目录下（脚本里写死了相对关系）：

```text
/2026aicompetition/workspace/dcs/
├── glioma_track4/             ← 算法工程：数据管线 + 一体化训练 + 全部脚本（主力）
├── glioma_goals/              ← 训练工程：六目标独立训练（兜底路线 B）
├── Glioma_recognition-main/   ← 提交工程：镜像内推理服务
└── CLOUD_USAGE_GUIDE.md       ← 本文
```

```bash
cd /2026aicompetition/workspace/dcs
ls -d */                                  # 期望看到上面三个工程目录
chmod +x glioma_track4/scripts/*.sh Glioma_recognition-main/start.sh 2>/dev/null
```

**平台路径约定**（不要改）：

| 用途 | 路径 |
|---|---|
| 训练集 | `/2026aicompetition/datasets/training/annotation` |
| 官方验证集 | `/2026aicompetition/datasets/verification`（其下是 `original/`） |
| 权重（提交读这里） | `/2026aicompetition/workspace/checkpoint/<goal>/` |
| 答案 | `/2026aicompetition/workspace/answer/<evaluation_id>/` |
| 日志 | `/2026aicompetition/workspace/logs` |

> 数据集不是天然可见的：创建容器时要在「存储与数据服务」里勾选训练/验证影像数据集，
> 没勾选的话目录压根不存在。

---

## 2. 环境准备

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

# 平台容器镜像自带 torch → 用 system 模式，只补轻量包（nibabel/openpyxl/fastapi…）
bash scripts/00_setup_env.sh --mode system

# 云桌面禁网时用离线 wheel
# bash scripts/00_setup_env.sh --mode system --offline --wheels wheels
# 只体检不安装
# bash scripts/00_setup_env.sh --mode verify
```

自检期望：

```text
[check] python 3.x | torch 2.x | cuda 12.x | 可用 True
[check] 缺失依赖: 无 ✓
```

> 报 `Error: externally-managed-environment` 时脚本已自动加 `--break-system-packages`，
> 不需要手工处理。

容器里往往只有 `python3`，直接跑 `.py` 时用 `python3`；`.sh` 脚本会自己探测。

---

## 3. 数据根定位

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
python3 scripts/29_locate_dataset_root.py
```

末行会直接给结论，例如：

```text
结论：数据根 = /2026aicompetition/datasets/training/annotation（3255 例）
      export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
```

填 `.../training` 或 `.../training/annotation` **都能跑通**（前者自动下钻到 `annotation/`）。
唯一会被直接拦下的是 `/2026aicompetition/datasets`（多阶段父目录，会 `ValueError`）——这是预期行为。

---

## 4. 探针（训练集清单）

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation

bash scripts/01_probe.sh          # → data/manifest.json
```

报告里**必须先看这两个数，不正常就别往下走**：

| 字段 | 期望 | 含义 |
|---|---|---|
| `series_type_rows` | **> 0** | 读到了数据信息表 `annotation/SeriesType.xlsx`；**0 = 表没接上**（模态会全判成 `other`） |
| `modality_counts` | 出现 `t1c` / `flair` / `t2` / `t1` | 模态识别正常 |
| `mask_role_counts` | 出现 `core` / `peri` | 掩膜识别正常（分割/全量训练要用） |
| `label_field_counts` | 越多越好 | 结构化字段金标准命中数（目标三/四的监督信号） |

数据信息表就在数据集里（与病例目录同层），探针自动读、零配置。
真找不到时用 `export GLIOMA_LABELS_DIR=<含表的目录>` 显式指定。

---

## 5. 折划分（只有兜底路线 B 需要）

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/02_build_dataset.sh            # → data/folds.json
python3 scripts/24_verify_eval_split.py     # 期望「汇总：X/X 项通过；0 项 WARN」
```

- **顺序不能反**：先 `01_probe.sh` 再 `02_build_dataset.sh`。换数据根后不重跑 01 会触发合规闸门报错。
- 只删 `data/folds.json` 重建时，**不要 `rm -rf data/`** —— `data/modality_model.json` 是随代码入库的
  模态兜底模型，误删后要重新 clone 或 `python3 scripts/31_train_modality_model.py --root <数据根>`。

> 全量训练（路线 A）**不需要** `folds.json`。这一步只是为了给兜底路线 B 准备。

---

## 6. 官方验证集清单（全量训练的 val）

`03_train.sh full` 的验证集就来自 `data/manifest_val.json`，必须先产出：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
export VAL_ROOT=/2026aicompetition/datasets/verification
bash scripts/01_probe.sh --val              # → data/manifest_val.json
```

**⚠️ 硬性前提：验证集里必须有带掩膜的病例。**
全量模式的 val 只用带掩膜的病例算 Dice 来选 best，一例都没有时训练器会直接退出。
先自检：

```bash
python3 - <<'PY'
import json
m = json.load(open("data/manifest_val.json"))
cs = m["cases"]
print("验证集病例:", len(cs), "| 带掩膜:", sum(1 for c in cs if c.get("masks")))
PY
```

- `带掩膜 > 0` → 可以继续。
- `带掩膜 == 0` → 全量模式没有早停集，只能退回折内 val（`bash scripts/03_train.sh 0`），
  或让数据方补验证集掩膜。
- 另外：官方验证集实测**只有 `SeriesType.xlsx`、没有字段金标准表**，这是预期的，
  不要因为 `label_field_counts` 为空去满磁盘找表。

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
[trainer] tag=smoke_full fold=full train=<清单全部病例> val=<带掩膜验证集例数> steps/epoch=...
[trainer] 全量模式：train=xxxx（清单全部病例） val=xxx（官方验证集 .../manifest_val.json）
```

2 个 epoch 结束、`checkpoints/smoke_full/best.pth` 出现 = 链路通过。

> **冒烟权重绝不能留在提交路径**：`TAG_PREFIX=smoke` 让它落在 `checkpoints/smoke_full/`，
> `09_export_submission.sh` 的自动选择只挑 `g4_fold*` / `g4_full*`，不会误选它；
> 用完可 `rm -rf checkpoints/smoke_full`。

---

## 8. 正式训练：全量 20 epoch（路线 A）

`epochs` 在**配置文件**里（`03_train.sh` 只透传 `--config`，没有 `--epochs` 参数）。
工程已备好一份 20 epoch 的全量配置：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
export VAL_ROOT=/2026aicompetition/datasets/verification
export CACHE_DIR=/2026aicompetition/workspace/cache

# 可选但强烈建议：先建预处理缓存（省掉训练时反复重采样）
python3 scripts/13_build_cache.py --workers 8

# 启动（前台；想后台加 nohup ... &）
CONFIG=train20 bash scripts/03_train.sh full
#   → checkpoints/g4_full/best.pth（+ 每 save_every 轮写 last.pth）
```

启动日志要看这两行：

```text
[trainer] tag=g4_full fold=full train=<全部病例> val=<带掩膜验证例数> ...
[trainer] 全量模式：train=xxxx（清单全部病例） val=xxx（官方验证集 .../manifest_val.json）
```

**单卡注意事项**：

- **不要设 `CUDA_VISIBLE_DEVICES=1`** —— 单卡机器上会把唯一的卡隐藏掉。留空（默认 0）或显式 `=0`。
- `configs/train20.yaml` 里可直接调的旋钮：
  - `patch_size`（显存不足降到 `[80,80,80]`）、`batch_size`（1 或 2）
  - `num_workers`（CPU 核多可到 8）
  - `val_every`（验证集例数多时改成 2，每两轮验证一次，明显提速）
  - `epochs`（本文件就是 20；想换回 100 用 `CONFIG=train`）

**运行期间若要并行跑别的训练**：两条路线的写路径完全不重叠
（`glioma_track4/checkpoints/` vs `glioma_goals/<goal>/runs/`），可以共卡并行；
唯一风险是显存 —— 盯着 `nvidia-smi`，谁报 OOM 谁停。**别在别人训练时跑 `02_build_dataset.sh`**
（它会重写 `folds.json`）。

---

## 9. 收尾：评估 + 导出提交权重

训练结束后：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

SKIP_MOCK=1 FOLDS=full bash scripts/16_finalize.sh
```

`16_finalize.sh` 五步一条龙：

| 步骤 | 做什么 |
|---|---|
| 1/5 | **阈值标定**：用集成模型在全图上扫最优阈值并写回权重（推理直接用它，避免固定 0.5 掉点） |
| 2/5 | **全图评估**：接了验证集 → 评官方验证集（external 口径）；没接 → 回退折内 val（OOF/留一） |
| 3/5 | 目标一/二 + 重复影像评估 |
| 4/5 | 按规范 §5.2**导出提交权重**到 `checkpoint/` |
| 5/5 | Mock Competition + 提交前检查清单（`SKIP_MOCK=1` 可跳过 Mock） |

- `FOLDS=full` 会映射到 `checkpoints/g4_full`；全量权重**必须**接上验证集走 external 口径，
  没接上它会明确报错让你先 `export VAL_ROOT=... && bash scripts/01_probe.sh --val`（不会静默用错口径）。
- 最终指标看 `logs/eval_external_ens.log` 和 3/5 段的官方验证集输出。

单独复核导出的权重：

```bash
bash scripts/09_export_submission.sh --verify
ls -l /2026aicompetition/workspace/checkpoint/*/     # 六个 goal 目录都要有文件
```

期望目录：

```text
checkpoint/
├── goal1_authenticity/model.pt
├── goal2_stitched/model.pt
├── goal2_duplicate/encoder.pt
├── goal3_tumor/model.pt
├── goal4_diagnosis/model.pt
└── goal5_segmentation/core.pt (+flair.pt)     ← 多折时保留多个 .pt 供集成
```

> ⚠️ `checkpoint/` 是平台评分真正读取的目录，`--verify` 只查文件是否齐备、
> **查不出"这是 1~2 epoch 的冒烟权重"**。演练一律用独立目录：
> `WORKSPACE=/tmp/ws_rehearsal bash scripts/09_export_submission.sh smoke_full`。

---

## 10. 提交前自检与本地闭环

```bash
############ ① 契约测试（提交工程）############
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 -m pytest tests/ -q                       # 期望全部 passed

############ ② 真实权重跑一次本地推理 ############
export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
export COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint
python3 scripts/local_eval.py \
  --dataset /2026aicompetition/datasets/verification \
  --output  /2026aicompetition/workspace/answer/local-001 \
  --evaluation-id local-001

############ ③ 复现平台协议（期望 5 行 PASS）############
python3 scripts/mock_competition.py \
  --dataset   /2026aicompetition/datasets/verification \
  --workspace /2026aicompetition/workspace --timeout 3600

############ ④ 答案目录校验 ############
python3 scripts/validate_output.py \
  --dir /2026aicompetition/workspace/answer/local-001 \
  --expect "$(ls /2026aicompetition/workspace/answer/local-001 | grep -v jsonl | paste -sd, -)"

############ ⑤ 算法工程提交前总检查（期望 15 通过 / 0 失败）############
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/23_pre_submit_check.sh                # 加 --quick 跳过耗时项
```

> 本地（无官方数据）跑 `23_pre_submit_check.sh` 会看到 **15 通过 / 1 失败**，
> 失败项固定是「权重未导出」；容器内做完第 9 步后应转绿。

启动推理服务（镜像内由 `start.sh` 拉起）：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
CALLBACK_URL="<平台回调地址>" bash scripts/06_platform_serve.sh
```

---

## 11. 兜底路线 B（可选，六目标独立训练）

只想留一份"链路已验证"的权重、或想按目标单独调参时用。**与路线 A 二选一**。

```bash
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
| 产物 | `runs/<tag>_foldN/checkpoints/best.pth`（**每个 epoch 结束才写 best.pth**，没跑完一轮就停会没有任何权重） |
| 全量训练 | 原始代码**不支持**（只有折划分/val_ratio）；要"全部训练集"就用路线 A 的 `03_train.sh full` |
| 打印的 `special` | `special` 头的 BCE 损失值（检查级二分类：假人体 / 拼接），首值约 0.69 = 未训练头初值，属正常 |

> ⚠️ **两条路线的权重最后写同一个 `checkpoint/<goal>/model.pt`，谁后导出谁覆盖。**
> 要提交就只保留一条路线的导出结果，别来回盖。
> 另外文档口径有出入：`MASTER_GUIDE` 说两条路线都能提交，`CLOUD_DESKTOP_RUNBOOK` §6.4 说
> 单路权重会被推理侧 `_check_heads` 拒绝 —— 以路线 A 为准，若要用路线 B 务必先跑
> `bash scripts/23_pre_submit_check.sh` 实测。

---

## 12. 环境变量一览

建议存成文件，免得每次重设：

```bash
mkdir -p /2026aicompetition/workspace/common
cat > /2026aicompetition/workspace/common/env.sh <<'EOF'
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
export VAL_ROOT=/2026aicompetition/datasets/verification
export GLIOMA_DATASET_ROOT="$DATASET_ROOT"      # glioma_goals 用的名字
export CACHE_DIR=/2026aicompetition/workspace/cache
export WORKSPACE=/2026aicompetition/workspace
export GLIOMA_CHECKPOINT_ROOT="$WORKSPACE/checkpoint"
export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
EOF
source /2026aicompetition/workspace/common/env.sh
```

| 变量 | 作用 | 备注 |
|---|---|---|
| `DATASET_ROOT` | 训练集数据根（算法工程） | `.../training` 会自动下钻；`.../annotation` 更稳 |
| `VAL_ROOT` | 官方验证集根 | 全量训练的 val 来源；已写进 `configs/paths.yaml`，env 可覆盖 |
| `GLIOMA_DATASET_ROOT` | 数据根（六个 Goal 工程） | 与 `DATASET_ROOT` 同值 |
| `CACHE_DIR` | 预处理缓存 | 放私有存储，容器删了不丢；默认已指向 `$WORKSPACE/cache` |
| `WORKSPACE` | 私有存储根 | 日志自动切到 `$WORKSPACE/logs` |
| `GLIOMA_CHECKPOINT_ROOT` | 提交权重根 | 规范 §5.2 的 `checkpoint/<goal>/` |
| `GLIOMA_FOLDS` | 统一折划分文件 | 六个 Goal 共用同一份 |
| `COMPETITION_PIPELINE_FACTORY` | 推理插件工厂 | **不设 = 跑 Dummy 基线**（看起来"跑通了"其实没加载权重） |
| `GLIOMA_LABELS_DIR` | 标注表目录（**可选**） | 表就在数据集里时**不用设**；乱设会串表 |

---

## 13. 常见问题对照

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| ① | `ValueError: 数据根 ... 指向数据集父目录，其下是平台阶段目录 [...]` | 填到了 `/2026aicompetition/datasets` | 指到具体阶段（`.../training`）。**这是预期行为**，不是 bug |
| ② | 探针 / 训练扫到 **0 例** | 数据根高了一层，或影像存储没挂到实例 | 先 `python3 scripts/29_locate_dataset_root.py`；`training/` 下只有 `annotation/` 属正常（会自动下钻）；连 `annotation/` 都没有 → 回平台勾选数据集或重建实例 |
| ③ | `series_type_rows = 0`、模态全是 `other`、报「无任何可用序列」 | 没读到 `annotation/SeriesType.xlsx` | 确认数据根指向与表同层的那层；表在别处才 `export GLIOMA_LABELS_DIR=<目录>` 后重跑 01/02 |
| ④ | `folds.json 的 fold=0 与当前数据不匹配 → 退回 val_ratio` | 折划分来自另一个数据集 | 按「先 01 再 02」重建；`data/` 是拷来的先删 `manifest.json`/`folds.json` |
| ⑤ | `03_train.sh full` 报「验证集清单里没有一例带掩膜」 | 官方验证集没有掩膜 | 全量模式需要带掩膜的早停集。用折内 val：`bash scripts/03_train.sh 0` |
| ⑥ | `16_finalize.sh` 提示「未接入官方验证集 → 回退折内 val」 | 缺 `data/manifest_val.json`，或全量 tag 撞上折内口径 | `export VAL_ROOT=... && bash scripts/01_probe.sh --val`；全量权重**必须**走 external |
| ⑦ | `local_eval` 只跑 Dummy 基线，像"跑通了" | 没设 `COMPETITION_PIPELINE_FACTORY` | `export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline` |
| ⑧ | 提交后分数极低但权重"齐全" | `checkpoint/` 里是演练（1~2 epoch）权重，`--verify` 查不出来 | 演练一律 `WORKSPACE=/tmp/ws_rehearsal`；提交前用正式权重重新导出 |
| ⑨ | `23_pre_submit_check.sh` 卡在「权重未导出」 | 没跑导出 | `bash scripts/09_export_submission.sh` |
| ⑩ | 训练日志里 `special ≈ 0.69` 不动 | 该头没有监督信号（batch 缺 `special_target`） | 属链路问题，看 `[loss][告警]`；正常收敛时它会持续下降 |
| ⑪ | `CUDA out of memory` | patch/batch 太大，或与其它训练共卡 | 降 `patch_size` / `batch_size` / `val_every`；共卡时保证显存余量 |
| ⑫ | 单卡上设了 `CUDA_VISIBLE_DEVICES=1` 后 `torch.cuda.is_available()` 变 False | 单卡机器只有 0 号卡 | 留空（默认 0）或显式 `=0` |

---

## 14. 关键路径速查

```text
数据（训练集）   /2026aicompetition/datasets/training/annotation
数据（验证集）   /2026aicompetition/datasets/verification
提交工程         /2026aicompetition/workspace/dcs/Glioma_recognition-main
训练工程 A       /2026aicompetition/workspace/dcs/glioma_track4
训练工程 B       /2026aicompetition/workspace/dcs/glioma_goals
提交权重         /2026aicompetition/workspace/checkpoint/<goal>/
答案             /2026aicompetition/workspace/answer/<evaluation_id>/
日志             /2026aicompetition/workspace/logs
预处理缓存       /2026aicompetition/workspace/cache
```

| 配置文件 | 作用 |
|---|---|
| `glioma_track4/configs/paths.yaml` | 数据根 / 验证集根 / 清单 / 缓存 / 导出路径（env 可覆盖） |
| `glioma_track4/configs/train.yaml` | 默认训练配置（100 epoch） |
| `glioma_track4/configs/train20.yaml` | **全量训练 20 epoch**（本方案用这份） |
| `glioma_track4/configs/_smoke_train.yaml` | 2-epoch 冒烟（小模型小 patch，指标无意义） |
| `glioma_track4/configs/preprocess.yaml` | 通道顺序、1mm 公共网格、推理 patch/阈值 |
| `glioma_track4/configs/labels.yaml` | 14 个结构化字段定义 |
| `glioma_goals/goal*/config.yaml` | 六个 Goal 各自的超参（只影响路线 B） |
