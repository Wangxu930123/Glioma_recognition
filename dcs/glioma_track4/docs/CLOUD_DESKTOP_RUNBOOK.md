# 云电脑操作手册：克隆 → 目录搬迁 → 数据根 → 训练/推理

> **本文件位置**
>
> - 克隆后未搬迁时：`/2026aicompetition/workspace/dcs/GliomaRecognition/dcs/glioma_track4/docs/CLOUD_DESKTOP_RUNBOOK.md`
> - 搬迁后：`/2026aicompetition/workspace/dcs/glioma_track4/docs/CLOUD_DESKTOP_RUNBOOK.md`
>
> 全文命令都可以直接复制粘贴。凡是形如 `CLONE=...`、`DATASET_ROOT=...` 的赋值，
> 请先按你机器上的实际路径改一次。

## 0. 全流程速查

```text
① 克隆 + 把三个工程搬到 /2026aicompetition/workspace/dcs/     → 第 1 节
② 装依赖                                                      → 第 2 节
③ 定数据根（含 annotation / SeriesType.xlsx 的那一层）        → 第 3、4 节
④ 探针确认模态与掩膜都认得出（series_type_rows > 0）          → 第 4 节
⑤ 重建折划分（必须先 01 再 02）                               → 第 5 节
⑥ 六折训练（--fold 0）                                        → 第 6 节
⑦ 推理与提交自检                                              → 第 7 节
```

**五个工程脚本路径**（本手册假设已按第 1 节搬迁）：

| 工程 | 作用 | 关键入口 |
|---|---|---|
| `glioma_track4` | 算法工程：探针 / 折划分 / 训练 / 评估 / 导出权重 | `scripts/01_probe.sh`、`scripts/02_build_dataset.sh`、`scripts/03_train.sh`、`scripts/16_finalize.sh` |
| `glioma_goals` | 六个 Goal 的独立训练入口 | `python smoke_all_goals.py`、`goal*/train.py` |
| `Glioma_recognition-main` | 提交工程：HTTP 服务 + 推理 + 写答案 + 校验 | `start.sh`、`scripts/local_eval.py`、`scripts/mock_competition.py` |

## 1. 克隆后把三个工程搬到 `dcs/` 下

`git clone` 出来的层级通常会多套一层（`GliomaRecognition/dcs/<工程>`），
而所有脚本都假定三个工程**并排**在 `/2026aicompetition/workspace/dcs/` 下。

### 1.1 先看清克隆出来的层级

```bash
ls -la /2026aicompetition/workspace/dcs/

# 三个工程实际在哪一层（不同平台嵌套层数可能不同）
find /2026aicompetition/workspace/dcs -maxdepth 4 -type d \
     \( -name glioma_track4 -o -name glioma_goals -o -name "Glioma_recognition*" \) 2>/dev/null
```

### 1.2 设定克隆根并核对

```bash
# ← 按 1.1 的输出改这一行（到"含三个工程目录的那一层"为止）
CLONE=/2026aicompetition/workspace/dcs/GliomaRecognition/dcs

ls "$CLONE"     # 期望看到：glioma_goals  glioma_track4  Glioma_recognition-main
```

### 1.3 搬迁（**推荐：旧目录改名备份，不直接删**）

旧目录里可能有权重（`checkpoints/`）、训练产物（`runs/`）、日志；
直接 `rm -rf` 会一并消失，而它们往往是你唯一的一份。

```bash
cd /2026aicompetition/workspace/dcs
TS=$(date +%Y%m%d_%H%M%S)

for p in glioma_goals glioma_track4 Glioma_recognition-main; do
  if [ -d "$p" ]; then
    mv "$p" "bak_${TS}_$p" && echo "旧 $p  →  bak_${TS}_$p"
  fi
  if [ -d "$CLONE/$p" ]; then
    mv "$CLONE/$p" ./ && echo "新 $p  就位"
  else
    echo "⚠️  $CLONE/$p 不存在，请核对目录名"
  fi
done

ls -d */ | sed 's/^/  /'
```

确认新工程能跑通后（第 3~6 节），再删备份：

```bash
rm -rf /2026aicompetition/workspace/dcs/bak_${TS}_*     # ⚠️ 确认无用再执行
```

### 1.4 如果你就是要"先删再移"

```bash
CLONE=/2026aicompetition/workspace/dcs/GliomaRecognition/dcs

rm -rf /2026aicompetition/workspace/dcs/glioma_goals
rm -rf /2026aicompetition/workspace/dcs/glioma_track4
rm -rf /2026aicompetition/workspace/dcs/Glioma_recognition-main

mv "$CLONE/glioma_goals" "$CLONE/glioma_track4" "$CLONE/Glioma_recognition-main" \
   /2026aicompetition/workspace/dcs/

ls -d /2026aicompetition/workspace/dcs/*/ | sed 's/^/  /'
```

### 1.5 搬迁后必须做的两件事

```bash
cd /2026aicompetition/workspace/dcs

# ① 脚本可执行位（git 有时不带可执行位）
chmod +x glioma_track4/scripts/*.sh Glioma_recognition-main/start.sh 2>/dev/null

# ② 清掉"从别处拷来的"旧清单与旧折划分
#    data/ 在 .gitignore 里，本来就不该跟着 git 走；若你的目录是从别处拷的，
#    里面会残留**另一个数据集**的 manifest.json / folds.json。
ls -la glioma_track4/data/ 2>/dev/null
rm -f glioma_track4/data/manifest.json glioma_track4/data/folds.json
```

> 为什么必须删：这两个文件里存的是**病例号和本机绝对路径**。
> 换数据根后继续用它们，不会报错，只会让折划分静默失效
> （详见第 5 节与第 8 节的报错 ④）。

## 2. 装依赖

```bash
cd /2026aicompetition/workspace/dcs

pip install -r Glioma_recognition-main/requirements.txt     # 推理/提交工程
pip install -r glioma_track4/requirements.txt               # 算法工程

# 官方数据要读 SeriesType.xlsx —— 训练侧的模态来源，缺了会报"无任何可用序列"
pip install openpyxl
```

报 `Error: externally-managed-environment` 时加 `--break-system-packages`，
或用平台镜像自带的 torch（本仓库的 requirements 不重复安装 torch）。

## 3. 数据根：定位与三个坑

### 3.1 平台挂载结构

```text
/2026aicompetition/
├── datasets/            ← 公共数据集（只读）
│   ├── training/        ← 官方训练集
│   ├── evaluation_first/ / evaluation_second/ / evaluation_finals/
│   └── verification/
├── public_models/       ← 公共模型（只读）
└── workspace/           ← 我的存储（读写 + 持久）
    ├── answer/  train_model/  common/
    ├── checkpoint/      ← 规范 §5.2：提交用权重
    └── dcs/             ← 代码 + 数据清单
```

> 数据集**不是天然可见的**：创建容器实例时要在「存储与数据服务」里勾选，
> 未勾选的存储就是个不存在的目录。

### 3.2 一条命令定数据根

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
python scripts/29_locate_dataset_root.py
```

输出三段：① 顶层挂载点 → ② 哪里有 NIfTI → ③ 各候选根的病例数，最后直接给结论：

```text
结论：数据根 = /2026aicompetition/datasets/training/annotation（3255 例）
      export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
```

**硬要求**：数据根必须精确到"含 `<检查号>/` 目录的那一层"，
并且该层要有与检查号同级的 **`SeriesType.xlsx`**。

### 3.3 三个坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `ValueError: 数据根 /2026aicompetition/datasets 指向数据集父目录，其下是平台阶段目录 [...]` | 数据根停在了父目录，阶段名会被当成检查号 | 指到具体阶段目录（如 `.../datasets/training`）；父目录下只有一个阶段时会自动下钻 |
| `training/` 下只有 `annotation/` | 实例只挂了标注那份存储 | 回「存储与数据服务」勾选训练影像数据集，或重建实例——不是代码问题 |
| 扫出 3255 例，但每个病例都挑不出模态（`无任何可用序列`） | UID 命名的序列认不出模态，缺 `SeriesType.xlsx` | 见第 4 节 |

## 4. 探针：确认模态与掩膜都认得出

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation   # ← 换成 3.2 的结论

bash scripts/01_probe.sh
```

看两个数（**它们不正常就别往下走**）：

| 报告字段 | 期望 | 含义 |
|---|---|---|
| `series_type_rows` | **> 0** | 读到 `SeriesType.xlsx` 的映射条数；0 = 类型表不在数据根那一层 |
| `modality_counts` | 出现 `t1c` / `flair` / `t2` / `t1` | 模态识别正常；只有 `other` = 全是 UID 命名、没读到类型表 |
| `mask_role_counts` | 出现 `core` / `peri` | 掩膜识别正常 |
| `label_field_counts` | 越多越好 | 结构化字段金标准命中数；0 = 没找到金标准表 |

找不到类型表时先定位它：

```bash
find /2026aicompetition/datasets -maxdepth 4 -name "SeriesType.xlsx"
```

它所在的层就是数据根（必须与 `<检查号>/` 同级）。

**背景**：训练侧原先只能从**目录名**猜模态。本地模拟集目录名是 `flair_0000` 所以一直正常；
官方数据的目录名是 DICOM UID（`1.2.826.0.1...`），任何关键词都命中不了，于是
"病例数正常、却一例都没有可用序列"。现已支持
`SeriesType.xlsx → 同名 .json sidecar → 目录名` 三级取值，两条训练路径都已接入。

## 5. 折划分：换数据必须重建（先 01，再 02）

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation

bash scripts/01_probe.sh           # ① 重建清单（写 manifest.json）
bash scripts/02_build_dataset.sh   # ② 重建折划分（写 folds.json）
```

`02` 会先打印清单来源，确认它就是你当前的数据根：

```text
[02] 清单来源：official/train_v1  数据根：/2026aicompetition/datasets/training/annotation
[02] 病例 3255 例（含标注/掩码 N）
  fold0: train 2604 / val 651
```

### 验证划分正确（**必须看**）

```bash
python scripts/24_verify_eval_split.py
```

期望：`汇总：X/X 项通过；0 项 WARN`。
出现下面这类 WARN 说明折划分本身是坏的（旧版实现的缺陷），要重建：

```text
WARN: 各折 val 存在重复（旧版 build_folds 缺陷）  3 例重复：[...]
WARN: 有病例从未进入任何折 val（无法获得 OOF）    2 例：[...]
```

### 为什么必须"先 01 再 02"

`folds.json` 里存的是**病例号**。如果拿 A 数据集的清单去给 B 数据集建折：

- 折里的病例号在 B 里一个都不存在 → 训练时每折的 val 都是空的；
- 旧版 `build_folds` 只比较**折数**，折数相同就原样返回旧划分 →
  `02` 打印的是新清单的病例数（**看着像重建成功了**），折却还是旧的。

现已加两道闸门：`02` 会校验清单与本机数据源是否一致；`build_folds`
复用旧划分前会校验它是否覆盖当前全部病例，不覆盖就重建并说明原因。

## 6. 训练：六个 Goal

### 6.1 单折冒烟（先确认能跑通）

```bash
cd /2026aicompetition/workspace/dcs/glioma_goals

# 数据集自检（CPU，秒级）：六个 Goal 都能建出样本
python smoke_all_goals.py --datasets-only --data "$DATASET_ROOT" --limit 8

# 真实的 1 epoch 训练自检（GPU，几分钟；会核对权重里的 cls_spec）
python smoke_all_goals.py --data "$DATASET_ROOT" --limit 8 --epochs 1
```

### 6.2 按折训练

```bash
cd /2026aicompetition/workspace/dcs/glioma_goals

for g in goal1_authenticity goal2_stitched goal2_duplicate \
         goal3_tumor goal4_diagnosis goal5_segmentation; do
  (cd $g && python train.py --tag exp1 --fold 0 --data "$DATASET_ROOT")
done
```

- `--fold N`：使用 `folds.json` 的第 N 折；**折号会自动进产物目录名** →
  `runs/exp1_fold0/`，四个折互不覆盖。
- 折号写错会**当场报错**并列出可用折（不会静默退回按比例划分）：

```text
--fold 9 不在 .../data/folds.json 的折里：可用 ['0','1','2','3','4']。
折划分由算法工程产出（bash scripts/02_build_dataset.sh）
```

### 6.3 算法工程侧的训练（多折）

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/03_train.sh all 4          # 4 折并行
bash scripts/16_finalize.sh             # 阈值标定 + 评估 + 导出权重
```

## 7. 推理与提交自检

### 7.1 契约测试（改完代码先跑）

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python -m pytest tests/ -q              # 期望全部 passed
```

### 7.2 本地跑一次完整推理（用真实权重）

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main

export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
export COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint

python scripts/local_eval.py \
  --dataset /2026aicompetition/datasets/verification \
  --output  /2026aicompetition/workspace/answer/local-001 \
  --evaluation-id local-001
```

### 7.3 复现平台协议（/health → /call → 后台 → 回调 → 校验）

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main

COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline \
COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint \
python scripts/mock_competition.py \
  --dataset  /2026aicompetition/datasets/verification \
  --workspace /2026aicompetition/workspace \
  --timeout 3600
```

期望 5 行 `PASS`：`Health` / `Call response time` / `Background execution` /
`Output and NIfTI validation` / `Callback`。

> `--dataset` 指向真实评测目录时**必须放大 `--timeout`**，
> 否则回调等待会先于推理结束而超时（单例实测约 165 秒）。

### 7.4 单独校验已产出的答案目录

```bash
python scripts/validate_output.py \
  --dir /2026aicompetition/workspace/answer/local-001 \
  --expect "$(ls /2026aicompetition/workspace/answer/local-001 | grep -v jsonl | paste -sd, -)"
```

### 7.5 导出提交权重（规范 §5.2 路径）

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/09_export_submission.sh            # 或 bash scripts/17_reset_for_official.sh 后按提示走
bash scripts/09_export_submission.sh --verify   # 复核
```

### 7.6 提交前总检查

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/23_pre_submit_check.sh             # 加 --quick 跳过耗时项
```

## 8. 已知报错对照表

| # | 报错 / 现象 | 原因 | 处理 |
|---|---|---|---|
| ① | `ValueError: 数据根 ... 指向数据集父目录，其下是平台阶段目录 [...]` | 数据根填成了父目录 | 用 `scripts/29_locate_dataset_root.py` 重定；指到具体阶段目录 |
| ② | `training/` 下只有 `annotation/`，扫不到病例 | 影像那份存储没挂到实例 | 实例的「存储与数据服务」勾选训练影像数据集 |
| ③ | `ValueError: study '...' 无任何可用序列`（病例数却正常） | 序列是 UID 命名，缺 `SeriesType.xlsx` | 已支持类型表/sidecar；确认数据根那层有 `SeriesType.xlsx`（第 4 节） |
| ④ | `folds.json 的 fold=0 与当前数据不匹配（train=0 val=0）→ 退回 val_ratio` | 折划分来自另一个数据集 | 按第 5 节"先 01 再 02"重建；`data/` 是从别处拷来的话先删 `manifest.json`/`folds.json` |
| ⑤ | `train.py: error: unrecognized arguments: --fold 0` | 旧版入口没有该参数 | 已支持：六个 `train.py` + 生成器模板都加了 `--fold`（产物落 `runs/<tag>_foldN/`） |
| ⑥ | 六个 Goal 的验证集不一致 / 指标不可比 | 某个 Goal 没读到 `folds.json`，自退回按比例划分 | 跑 `python scripts/25_verify_tasks_integration.py`，看"六个 Goal 的验证集完全一致"是否通过 |
| ⑦ | `local_eval` 只跑 Dummy 基线，像"跑通了" | 旧版直接构造 Settings，忽略了 `COMPETITION_PIPELINE_FACTORY` | 已修（改用 `Settings.from_env()`）；确认环境变量已 export |
| ⑧ | `config.yaml` 里的其它 Goal 参数不生效 | 参数是写在**副本** `tr` 上，`build_datasets` 读的是 `cfg` | 已修（回写 `cfg["train"]`）；自定义参数时注意同一坑 |
| ⑨ | `找不到数据集根目录；请用 --data 指定…`（明明 export 了数据根） | 两个工程历史上用不同的变量名：算法工程 `DATASET_ROOT`、研发侧 `GLIOMA_DATASET_ROOT` | 已支持互相兼容（`GLIOMA_DATASET_ROOT` / `DATASET_ROOT` / `DATASET_PATH` 任一均可），建议按 §10 两个都设 |
| ⑩ | 训练日志出现 `special: 0.0` / `embed: 0.0` | 损失已收敛（小数据集上很快被"背下来"），不是信号断裂 | 首行若为 `special≈0.69`（未训练头初值）即链路正常；缺监督信号时会打印 `[loss][告警]` |

## 9. 一键自检清单

```bash
cd /2026aicompetition/workspace/dcs
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation   # ← 换成 3.2 的结论

echo "① 团队契约测试"; (cd Glioma_recognition-main && python -m pytest tests/ -q 2>&1 | tail -1)
echo "② 对接断言（含平台布局/类型表/折划分/--fold）"; (cd glioma_track4 && python scripts/25_verify_tasks_integration.py 2>&1 | tail -2)
echo "③ P0 修复断言"; (cd glioma_track4 && python scripts/21_verify_fixes.py 2>&1 | grep 汇总)
echo "④ 折划分正确性"; (cd glioma_track4 && python scripts/24_verify_eval_split.py 2>&1 | grep 汇总)
echo "⑤ 六个 Goal 数据集"; (cd glioma_goals && python smoke_all_goals.py --datasets-only --data "$DATASET_ROOT" --limit 8 2>&1 | tail -1)
echo "⑥ 数据根一览"; (cd glioma_track4 && python scripts/29_locate_dataset_root.py --max-depth 4 2>&1 | tail -3)
```

期望：① 全 passed ② 全部通过 ③ 16/16 ④ 0 项 WARN ⑤ 通过 6/6。

## 10. 环境变量一览

```bash
# 建议写进一个 env 文件，免得每次重设
mkdir -p /2026aicompetition/workspace/common
cat > /2026aicompetition/workspace/common/env.sh <<'EOF'
# 数据根：两个工程历史上用了不同的变量名，这里**两个都设**，免得互相踩
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation   # ← 换成 3.2 的结论
export GLIOMA_DATASET_ROOT="$DATASET_ROOT"        # glioma_goals 用的名字

export GLIOMA_FOLDS=/2026aicompetition/workspace/dcs/glioma_track4/data/folds.json
export WORKSPACE=/2026aicompetition/workspace
export CACHE_DIR=/2026aicompetition/workspace/cache     # 缓存放私有存储，容器删了不丢
export GLIOMA_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint
export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
EOF
source /2026aicompetition/workspace/common/env.sh
```

| 变量 | 作用 | 备注 |
|---|---|---|
| `DATASET_ROOT` | 数据根（算法工程） | 必须精确到含 `<检查号>/` 与 `SeriesType.xlsx` 的那一层 |
| `GLIOMA_DATASET_ROOT` | 数据根（研发侧六个 Goal） | 与上一行同值；两个名字都已支持，设一个也行 |
| `GLIOMA_FOLDS` | 统一折划分文件 | 六个 Goal 共用同一份，否则指标不可比 |
| `WORKSPACE` | 私有存储根 | 容器删除后数据会清除，产物必须放这里 |
| `CACHE_DIR` | 预处理缓存 | 放私有存储，避免重建 |
| `GLIOMA_CHECKPOINT_ROOT` | 提交权重的根目录 | 规范 §5.2：`checkpoint/<goal>/...` |
| `COMPETITION_PIPELINE_FACTORY` | 推理插件工厂 | 不设 = 跑 Dummy 基线 |
| `GLIOMA_LOADER_TOLERANT` | 推理加载容错 | 平台启动脚本默认开启：单个脏文件不让整批失败 |

## 附：排查文档索引

| 文档 | 内容 |
|---|---|
| `glioma_track4/docs/CLOUD_DESKTOP_RUNBOOK.md` | **本文件**：克隆/搬迁/训练/推理全流程 |
| `glioma_track4/docs/DATASET_ROOT_TROUBLESHOOT.md` | 数据根排查（含"只有 annotation""无任何可用序列"） |
| `glioma_track4/docs/PLATFORM_GUIDE.md` | 平台使用指南（容器、存储、依赖） |
| `glioma_track4/README.md` | 算法工程说明（§4.0 平台数据目录） |
| `Glioma_recognition-main/README.md` | 提交工程说明（平台数据目录、本地演练） |
