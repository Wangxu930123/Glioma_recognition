# 赛道四 · 三工程运行总说明

> **版本 v2（2026-09-23）**：修复「数据根填高一层 → 静默扫到 0 例」；两个训练工程的数据根默认值已统一为平台路径。
> 适用工程：`Glioma_recognition-main`（提交）、`glioma_track4`（数据管线 + 一体化训练）、`glioma_goals`（六目标并行训练）

**这份文档放在哪里看**（`glioma_track4/docs/`，四份**不重复**，按需查）

| 什么时候看 | 看哪份 | 它独有的内容 |
|---|---|---|
| 搞清楚整体怎么跑（默认） | **本文** | 三工程串联、数据契约、本次修复、容器内全链路命令、验证清单 |
| 第一次上机：点哪里、账号怎么弄 | `PLATFORM_GUIDE.md` | 赛事平台登录/申资源、云桌面客户端安装、创建容器实例、SSH 与克隆（含 Token 兜底）、创建测评容器并发起评测 |
| 目录要搬迁 / 探针字段为空 / 换数据重建折划分 | `CLOUD_DESKTOP_RUNBOOK.md` | 克隆层级不固定时的搬迁、官方标注表的定位与逐项核对、`label_field_counts` 为空的三级定位、**先 01 再 02** 的折划分规矩、报错对照表 |
| 报 `ValueError: 数据根 ... 指向数据集父目录` | `DATASET_ROOT_TROUBLESHOOT.md` | 数据根逐层定位与判定（单点排查，11KB） |
| 要一步步照抄命令跑完算法工程全流程 | `TRACK4_RUNBOOK.md` | 变量表、每步命令与**期望输出**、验收判据、"验证集没有金标准能不能做全量训练"、报错对照表、提交前检查清单 |

> 组委会的《赛道四_自建模型组_比赛背景与开发规范.md》《…_系统架构与协作规范.md》是**权威口径**
> （提交约束、服务接口、权重路径），本文是执行手册；两者冲突时以规范为准（这两份不在本仓库内）。

---

## 0. 一分钟速览

1. 三个工程**不合并**，彼此只有两条接口：**数据布局**（平台给的）与**权重目录**（规范 §5.2 规定的）。
2. 权重落地契约只有一个：`<workspace>/checkpoint/<goal>/model.pt`（目标二-B 是 `encoder.pt`；目标五多折保留多个 `.pt`）。提交工程在**服务启动时**读它，不关心是哪条训练路线产出的。
3. 数据根**填 `.../datasets/training` 或 `.../datasets/training/annotation` 都能跑通**：前者会自动下钻到后者并打印 `[probe][告警]`（v2 新增，见第 3 节）。
4. 唯一会被直接拦下的是 `/2026aicompetition/datasets` 这类**多阶段父目录**——它会 `ValueError` 报错，不会猜。

---

## 1. 三工程分工

```text
        ┌──────────────────────── 平台数据（只读） ────────────────────────┐
        │  /2026aicompetition/datasets/training/annotation/               │
        │    <检查号>/<序列UID>/<序列UID>.nii.gz（表在工作区）  │
        └──────────┬────────────────────────────────────┬────────────────┘
                   │                                    │
     ┌─────────────▼─────────────┐      ┌───────────────▼─────────────┐
     │  glioma_track4            │      │  glioma_goals               │
     │  数据管线 + 一体化训练      │      │  六目标独立训练（并行）      │
     │  探针→折划分→缓存→训练→导出 │      │  每人一个 goal、一张卡       │
     └─────────────┬─────────────┘      └───────────────┬─────────────┘
                   │  09_export_submission.sh           │
                   └───────────────┬────────────────────┘
                                   ▼
             /2026aicompetition/workspace/checkpoint/
             ├── goal1_authenticity/model.pt    ├── goal3_tumor/model.pt
             ├── goal2_stitched/model.pt        ├── goal4_diagnosis/model.pt
             ├── goal2_duplicate/encoder.pt     └── goal5_segmentation/<tag>.pt
                                   │
                                   ▼
     ┌──────────────────────────────────────────────────────────┐
     │  Glioma_recognition-main（提交工程，打进镜像）             │
     │  app.server:app ← uvicorn --host 0.0.0.0 --port 8000      │
     │  收到 /call → 读数据集 → 跑六目标 → 写 answer/             │
     └──────────────────────────────────────────────────────────┘
```

| 工程 | 角色 | 关键入口 | 本次改动 |
|---|---|---|---|
| `Glioma_recognition-main` | 提交工程，镜像内常驻服务 | `start.sh` → `app.server:app` | **无**（已确认可正常提交） |
| `glioma_track4` | 数据管线 + 一体化训练 + 提交前验证 | `scripts/01~29`、`scripts/23_pre_submit_check.sh` | 数据根下钻 + 文档 |
| `glioma_goals` | 六目标并行训练（五个人分工） | `goal*/train.py`、`smoke_all_goals.py` | 数据根下钻 + 六份配置 |

**六目标与官方任务的对应**（`glioma_goals`）：

| Goal 目录 | 官方任务 | 交付权重文件名 |
|---|---|---|
| `goal1_authenticity` | ① 影像真实性（假人体 / 非人体） | `model.pt` |
| `goal2_stitched` | ②-A 拼接影像检测 | `model.pt` |
| `goal2_duplicate` | ②-B 重复影像检测（配对） | `encoder.pt` |
| `goal3_tumor` | ③ 病灶识别（胶质瘤 vs 非肿瘤） | `model.pt` |
| `goal4_diagnosis` | ④ 辅助诊断（14 个结构化字段） | `model.pt` |
| `goal5_segmentation` | ⑤ 分割（T1C 核心区 + FLAIR 周边区） | `<tag>.pt`（多折集成） |

---

## 2. 平台数据长什么样（官方契约）

### 2.1 官方工程给的硬约定

从组委会参考工程 `AIRecongition` 逐字抽出（`configs/config.yaml` + `src/data/paths.py`）：

```yaml
annotation_root: /2026aicompetition/datasets/training/annotation
labels_dir:      ./labels          # 5 张标注表所在目录，可用配置覆盖
```

```python
# src/data/paths.py —— PathResolver(annotation_root)
真例    : root/<AccessionNumber>/<SeriesUid>/<SeriesUid>.nii.gz
异常影像: root/fake/<AccessionNumber>/<SeriesUid>/<SeriesUid>.nii.gz
          root/compositing/...      root/duplicate/...
掩膜    : root/<AccessionNumber>/<SeriesUid>/<mask_name>
```

**关键点：真例与异常影像在 `annotation_root` 里是同级兄弟**，都在这一层之下。

### 2.2 云桌面实测结论

`CLOUD_DESKTOP_RUNBOOK.md` §3.2 记录的是**实测**结果（不是推断）：

```text
结论：数据根 = /2026aicompetition/datasets/training/annotation（3255 例）
```

实测到的两件事，都容易踩：

| 现象 | 说明 |
|---|---|
| `training/` 下**只有** `annotation/` | 影像不直接在 `training/` 里，第一次很容易指错一层 |
| **数据信息 `SeriesType.xlsx` 与病例目录同层**（`annotation/` 下） | 它**不是**数据根的下一层、也不在团队工作区；表里 `SeriesType` 是模态的唯一可靠来源。以前只认工作区那份 `3_serieslabel.xlsx`、且只在少数目录里找，于是"表明明在磁盘上却读不到" → 模态全是 `other` → 取数时报「无任何可用序列」。现已统一按候选目录（`$GLIOMA_LABELS_DIR` → `<工程>/labels` → `$WORKSPACE` 下 3 层 → 数据根/父/祖父 → 像标注容器的子目录）优先搜数据集里的 `SeriesType.xlsx` |

### 2.3 数据路径与数据信息路径（平台实测）

**★ 数据路径（影像）**：`<阶段>/annotation/<32位检查号>/<2.25.* 序列UID>/<序列UID>.nii.gz`
**★ 数据信息路径**：`<阶段>/annotation/SeriesType.xlsx`（与病例目录**同层**）

```text
/2026aicompetition/datasets/
├── training/                             ← 训练集（数据根填这一层，或直接填 annotation/）
│   └── annotation/                       ← annotation_root，最稳
│       ├── SeriesType.xlsx               ★ 数据信息：序列类型（T1 / T1CE（增强）/ T2-Flair / T2WI / 其他）
│       ├── 脑胶质瘤标注结果-训练集.xlsx    ★ 数据信息：结构化字段金标准（训练集才有）
│       ├── <32位检查号>/<2.25.* 序列UID>/<序列UID>.nii.gz     ★ 影像数据
│       └── {fake,compositing,duplicate}/<检查号>/...
├── verification/                         ← 验证集（结构与 training 同）
│   └── annotation/
│       ├── SeriesType.xlsx               ★ 数据信息（验证集也有）
│       └── <32位检查号>/<2.25.* 序列UID>/<序列UID>.nii.gz
├── evaluation_first/                     ← 评测阶段：平台通过请求体 input.dataset_path 传入
├── evaluation_second/                    ←   （SeriesType.xlsx 随测试数据一起下发，路径规则同上）
└── evaluation_finals/
```

| 用途 | 路径（以 `training` 为例） |
|---|---|
| **影像数据** | `/2026aicompetition/datasets/training/annotation/<32位检查号>/<2.25.* 序列UID>/<序列UID>.nii.gz` |
| **数据信息（序列类型/模态）** | `/2026aicompetition/datasets/training/annotation/SeriesType.xlsx` |
| 数据信息的列 / 取值 | 列 `AccessionNumber` + `SeriesUid` + `SeriesType`；取值 **5 类**：`T1`、`T1CE（增强）`、`T2-Flair`、`T2WI`、`其他` |
| 取值 → 通道键 | `T1`→`t1`，`T1CE（增强）`→`t1c`，`T2-Flair`→`flair`，`T2WI`→`t2`，`其他`→**排除**（不是模态，不交给体素模型猜） |
| 评测集 | 同样有 `SeriesType.xlsx`，**正式测试时与测试数据一起下发**；路径由 `input.dataset_path` 给出，不需要猜 |

> ⚠️ **`1_abnormal.xlsx` / `2_duplicate.xlsx` / `3_serieslabel.xlsx` / `4_masklabel.xlsx` /
> `5_characteristics.xlsx` 这 5 张表与赛道四数据集没有关系**（属工作区里另一个目标的产物）。
> 赛道四的数据信息就是数据集自带的 `SeriesType.xlsx`（加上训练集的
> `脑胶质瘤标注结果-训练集.xlsx` 字段金标准）。
> 代码保留对 `3_serieslabel.xlsx` 的查找**只为兼容兜底**：仅在数据集的 `SeriesType.xlsx`
> 读不到时才用，且**只补缺、不覆盖** —— 顺序反了会静默把 `T2WI` / `T2-Flair` 覆盖成
> 粗粒度的 `T2`，表现是"模态看着都认出来了、通道里却是错的对比度"。
> 详见 `DATASET_ROOT_TROUBLESHOOT.md`「病例数正常、却报无任何可用序列」。

> `evaluation_*` 的路径**不需要猜**：平台在 `/call` 请求里以 `input.dataset_path` 传入，提交工程直接用。

### 2.4 答案产物格式（提交工程负责，供核对）

```text
/2026aicompetition/workspace/answer/<evaluation_id>/
├── <AccessionNumber>/
│   ├── prediction.json                       # 逐病例的结构化结论
│   └── <SeriesUid>/
│       ├── <SeriesUid>_core.nii.gz           # 目标五分割：核心区
│       └── <SeriesUid>_abnormal.nii.gz       # 异常 / 假人体标记
└── duplicate_pairs.jsonl                     # 目标二-B 重复对（批级一份）
```

回调 payload：`{"request_id": ..., "evaluationId": ..., "predPath": ...}`。

---

## 3. 本次修复：数据根不再「填错一层就静默 0 例」

### 3.1 问题

两个训练工程的发现逻辑都是「**跳过** `annotation` 目录」，但**不会下钻**进去：

- 数据根填 `.../training`（该层只有 `annotation/`）→ 真例扫描 **0 例**；
- 而且**不报错**：探针照常出报告、训练照常启动，只是损失一直不动；
- 「自动下钻」此前**只实现在提交工程**（`data/loader.py::_resolve_dataset_root`），两个训练工程没有。

更危险的是 `glioma_goals` 六个 `config.yaml` 的 `data.root` **硬编码本机路径**
`/mnt/data_sdb/wangx/data/Brain_MRI/track4_sim`：容器里要么报错，要么万一同名路径存在就
**静默在非官方数据上训练**（违反数据源合规要求）。

### 3.2 修复后的行为

两个训练工程现在都实现了 `resolve_case_root()`，规则与提交工程一致：

| 数据根填成 | 行为 | 结果 |
|---|---|---|
| `.../datasets/training/annotation` | 本层就是病例层 | ✅ 直接用（推荐） |
| `.../datasets/training` | 下一层唯一候选是 `annotation/` | ✅ **自动下钻** + 打印 `[probe][告警]` |
| `.../datasets`（多阶段父目录） | 候选多个，**不猜** | ✅ `ValueError` 拦住，提示指到具体阶段 |
| 不存在的路径 | 原样返回 | ✅ 后续报错，不会静默 |

设计原则：**只在「下一层唯一候选」时下钻**。候选多于一个时，猜错阶段比直接失败更糟，所以宁可报错
（`.../datasets` 下若只有一个阶段目录，也会自动下钻）。

### 3.3 改动清单

**`glioma_track4`**

| 文件 | 改动 |
|---|---|
| `src/data/probe.py` | 新增 `resolve_case_root()`；接入 `scan_real()` 与 `probe()` |
| `configs/paths.yaml` | 修正误导注释（原文称「其下能看到 `annotation/{...}` 与各检查号目录」与实测不符） |
| `docs/CLOUD_DESKTOP_RUNBOOK.md` §3.2 | 「硬要求：必须精确到病例层」→ 放宽为「两层都能跑」 |
| `docs/DATASET_ROOT_TROUBLESHOOT.md` | 过时结论示例（`training` / 204 例）→ 实测值（`training/annotation` / 3255 例） |
| `docs/PLATFORM_GUIDE.md` | 修正「两种常见结构」（原 A/B 都对不上真实布局）+ 候选根补上 `annotation` |
| `README.md`、`scripts/17_reset_for_official.sh` | 同步数据根说明 |

**`glioma_goals`**

| 文件 | 改动 |
|---|---|
| `shared/data.py` | 新增同款 `resolve_case_root()`；接入 `discover_cases()` |
| 六个 `goal*/config.yaml` | `data.root`：本机路径 → `/2026aicompetition/datasets/training` |
| `gen_goals.py` | 生成模板同步（否则重新生成会退回本机路径） |

### 3.4 为什么默认值仍写 `.../training`

因为**默认值填上一层更稳**，配合下钻后两种平台布局都能覆盖：

- 真例直接在 `training/<检查号>/`（早期 204 例挂载形态）→ 本层就有病例目录，**不下钻**，命中；
- 真例在 `training/annotation/<检查号>/`（当前 3255 例形态）→ 自动下钻，命中。

反过来，默认值若写死 `.../training/annotation`，遇到第一种形态就**无解**（下钻只能向下）。

---

## 4. 容器内全链路命令（从零到提交）

> 在**云桌面容器**内执行。阶段 2 的两条训练路线**二选一**，权重都落到同一个契约目录。

### 阶段 0 · 环境

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/00_setup_env.sh                                # 依赖与目录准备

export DATASET_ROOT=/2026aicompetition/datasets/training     # 填这层即可，会自动下钻并告警
export CACHE_DIR=/2026aicompetition/workspace/cache          # 缓存放私有存储，容器删除不丢
```

### 阶段 1 · 数据（顺序不能换）

```bash
python scripts/29_locate_dataset_root.py      # ① 不确定数据根时先定位（打印候选根与病例数）
bash scripts/01_probe.sh                      # ② 探针 → data/manifest.json + 结构报告
bash scripts/02_build_dataset.sh              # ③ 折划分（含数据源合规闸门）
python scripts/13_build_cache.py --workers 8  # ④ 预处理缓存（可选，强烈建议）
```

> `1_abnormal` 的 `Label` 指错目录时（compositing/duplicate 的行被标成 `true`，官方 `series_path` 就拼不出路径），
> 在探针前先跑 **`bash scripts/33_fix_abnormal_labels.sh`**：自动接上团队工作区的 `labels/`（`export GLIOMA_LABELS_DIR`）、
> 用磁盘核对把人工补丁（如 `1_abnormal_wzh.xlsx`）合并回 `1_abnormal.xlsx`（命中率上升才写、自动备份），最后重跑探针。
> 只想看差别不写文件：加 `--dry-run`。
>
> 数据信息 `annotation/SeriesType.xlsx` 与影像同层，**探针自动读、零配置**；
> 工作区那份 `labels/3_serieslabel.xlsx`（与数据集无关）会自动在 `$WORKSPACE` 下 3 层内搜索
> （覆盖 `<workspace>/dcs/*/*/labels`，跳过 `cache`/`logs`），但**只在数据里那份读不到时才用**。
> 表里"检查号"列与磁盘病例目录名不一致时，自动按 `SeriesUid` 单键回退。
> 排查见 `docs/DATASET_ROOT_TROUBLESHOOT.md`。

**探针报告里必须先看这两个数，不正常就别往下走**：

| 字段 | 期望 | 含义 |
|---|---|---|
| `series_type_rows` | **> 0** | 读到了数据信息 `<阶段>/annotation/SeriesType.xlsx`（兜底 `labels/3_serieslabel.xlsx`）；为 0 = 都没找到（会自动搜数据根/父/祖父 + `$WORKSPACE` 下 3 层，找不到就 `export GLIOMA_LABELS_DIR=<含表的目录>` 再重跑探针） |
| `modality_counts` | 出现 `t1c` / `flair` / `t2` / `t1` | 模态识别正常；只有 `other` = 全是 UID 命名、没读到数据信息表 |

### 阶段 2 · 训练（二选一）

**路线 A —— `glioma_track4` 一体化训练**

```bash
bash scripts/03_train.sh all 4                # 4 折并行（4 张卡）
FOLDS="0 1 2 3" bash scripts/16_finalize.sh   # 收尾：评估 + 出提交物
```

**路线 B —— `glioma_goals` 六目标并行（五个人分工）**

```bash
cd /2026aicompetition/workspace/dcs/glioma_goals

# 先体检：只建数据集、不训练（秒级，无需 GPU）
python smoke_all_goals.py --datasets-only --data "$DATASET_ROOT" --limit 8

# 再冒烟：每个 Goal 跑 1 个 epoch，核对 best.pth 的 cls_spec 与分类头数量（需 GPU）
python smoke_all_goals.py --data "$DATASET_ROOT" --limit 8

# 正式训练：每人进自己目录、占一张卡
cd goal5_segmentation && CUDA_VISIBLE_DEVICES=0 python train.py --tag my_exp --data "$DATASET_ROOT"
```

### 阶段 3 · 导出权重（**提交前必做**）

```bash
# 路线 A
cd /2026aicompetition/workspace/dcs/glioma_track4 && bash scripts/09_export_submission.sh
# 路线 B
cd /2026aicompetition/workspace/dcs/glioma_goals && TAG=exp1 bash scripts/export_to_submission.sh

ls -l /2026aicompetition/workspace/checkpoint/*/     # 六个目录都要有权重
```

### 阶段 4 · 提交前自检（**必须全绿**）

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/23_pre_submit_check.sh          # 全量，约 5~10 分钟
bash scripts/23_pre_submit_check.sh --quick  # 快速版（跳过耗时项）
```

### 阶段 5 · 提交

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
# 按《开发规范》§5.2 打镜像 / 触发评测；服务由 start.sh 启动：
#   python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 1
```

> **从本地验证切回官方数据时**：先 `bash scripts/17_reset_for_official.sh` 归档本地产物（不删除），
> 再用官方数据重跑阶段 1。

---

## 5. 验证清单与预期结果

### 5.1 本地可跑（无需平台数据、无需 GPU）

| # | 命令 | 预期 |
|---|---|---|
| 1 | `python smoke_all_goals.py --datasets-only --data <真实布局数据根> --limit 8` | 六个 Goal 全 `[PASS]`，末行 `通过 6/6` |
| 2 | `bash scripts/23_pre_submit_check.sh --quick` | 语法 / 导入 / 契约断言全绿 |
| 3 | 附录 A 的下钻自检 | `下钻: True`、`护栏: ✓` |

### 5.2 容器内必跑

| # | 命令 | 预期 |
|---|---|---|
| 1 | `python scripts/29_locate_dataset_root.py` | 结论行给出数据根与病例数（**3255 例**量级） |
| 2 | `bash scripts/01_probe.sh` | `series_type_rows > 0`、`modality_counts` 有真实模态、manifest 写出 |
| 3 | `bash scripts/09_export_submission.sh` | `checkpoint/<goal>/` 六个目录齐全 |
| 4 | `bash scripts/23_pre_submit_check.sh` | **15 通过 / 0 失败** |

> 本地跑 `23_pre_submit_check.sh` 会看到 **15 通过 / 1 失败**，失败项固定是「权重未导出」——
> 因为本地没有官方数据可训。**这一项必须在容器内做完阶段 3 后转绿**，
> 它正是「忘了导权重就提交」的最后一道防线。

### 5.3 本次修复的实测结果

| 验证 | 结果 |
|---|---|
| 下钻自检（填高一层 / 精确层 / 多阶段父目录 / 不存在路径） | **7/7 通过**，护栏未被削弱 |
| `glioma_track4` 在平台式布局（根=`.../training`）跑 `01_probe.sh` | ✅ 自动下钻，扫到 **612 例**，manifest 正常 |
| `glioma_goals` 同布局跑 `smoke_all_goals.py --datasets-only` | ✅ **6/6 通过** |
| `glioma_track4` 全量 `23_pre_submit_check.sh` | ✅ **15 通过 / 1 失败**（唯一失败 = 权重未导出，属容器内步骤） |
| 运行期代码硬编码本机路径扫描（`src/`、`shared/`、`goal*/`） | ✅ 0 命中 |
| lint（改动的两个 `.py`） | ✅ 0 诊断 |

---

## 6. 权重交付链路（唯一的跨工程契约）

```text
glioma_goals/<goal>/runs/<tag>/checkpoints/best.pth      （路线 B）
glioma_track4/checkpoints/<tag>/best.pth                 （路线 A）
        │  09_export_submission.sh / export_to_submission.sh
        │  = 复制 + 校验元信息 + 改名到规范文件名
        ▼
<workspace>/checkpoint/<goal>/model.pt
        │  提交工程在**服务启动时**读取（tasks/<goal>/config.py 的 ckpt_rel）
        ▼
镜像内推理进程
```

- 权重**不进代码仓库**（规范 §5.2），统一放 `/2026aicompetition/workspace/checkpoint/`。
- 缺产物时导出脚本**非 0 退出并指出缺哪个**，不会静默跳过。
- 目标五多折时保留多个 `.pt`，提交工程会自动读取该目录下全部 `.pt` 做集成。

---

## 7. 常见坑与排障

| 现象 | 原因 | 处理 |
|---|---|---|
| 探针 / 训练扫到 **0 例** | 数据根填高了一层 | v2 已自动下钻并告警；若告警说钻错地方，用 `DATASET_ROOT` 显式指定。先跑 `python scripts/29_locate_dataset_root.py` |
| `ValueError: 数据根 ... 指向数据集父目录，其下是平台阶段目录 [...]` | 填到了 `/2026aicompetition/datasets` | 指到具体阶段（如 `.../datasets/training`）。**这是预期行为** |
| `training/` 下只有 `annotation/` | 实例只挂了标注那份存储 | v2 会自动下钻。若连 `annotation/` 都没有 → 回「存储与数据服务」勾选训练影像数据集或重建实例，**不是代码问题** |
| 扫出 3255 例但挑不出模态（`无任何可用序列`） | UID 命名认不出模态，没读到数据信息 `<阶段>/annotation/SeriesType.xlsx` | 报错**已自带自检**（表在哪 / 体素模型在不在）。先确认数据里那张表在（它**与病例目录同层**）；表在非常规位置才需 `export GLIOMA_LABELS_DIR=<含表的目录>` 后重跑 `bash scripts/01_probe.sh` + `bash scripts/02_build_dataset.sh`。详见 `docs/DATASET_ROOT_TROUBLESHOOT.md`「病例数正常、却报无任何可用序列」 |
| `label_field_counts = 0` 或 `official_label_files: {}` | 天坛参考实现那 5 张英文表**与赛道四数据集无关**、也不随数据集下发（它们躺在**团队工作区**里，如 `.../workspace/dcs/goal1and2/Goal1and2/labels/`）；数据集自带的是中文表头的 `annotation/脑胶质瘤标注结果-训练集.xlsx` | 先看 `label_field_counts`（中文表已覆盖）；确实要用英文表就 `export GLIOMA_LABELS_DIR=<那个 labels 目录>` 后重跑 `bash scripts/01_probe.sh` |
| `1_abnormal` 把 compositing/duplicate 的行标成 `true`（`Label` 指错目录 → 官方 `series_path` 拼不出路径） | 人工修补版（如 `1_abnormal_wzh.xlsx`）没合并回 `labels/1_abnormal.xlsx` | **`bash scripts/33_fix_abnormal_labels.sh`** 一条命令搞定（自动定位 → 按 `Label` 拼路径做磁盘核对 → 命中率上升才写回并备份 → 重跑探针）；要手动控制细节则用 `scripts/32_apply_abnormal_patch.py` |
| `02_build_dataset.sh` 报「清单与数据根不同类」 | 换了数据根却没重跑 `01_probe`（合规闸门） | 先 `bash scripts/01_probe.sh`；从本地切回官方数据先跑 `17_reset_for_official.sh` |
| `23_pre_submit_check.sh` 卡在「权重未导出」 | 没跑阶段 3 | `bash scripts/09_export_submission.sh` |
| `glioma_goals` 找不到数据根 | `--data` 与配置默认值都不对 | 优先级：`--data` > `GLIOMA_DATASET_ROOT` > `DATASET_ROOT` > `DATASET_PATH` > 配置默认值（现已是平台路径） |

---

## 8. 速查

### 8.1 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `DATASET_ROOT` | `glioma_track4` 数据根（覆盖 `raw.track4`） | `/2026aicompetition/datasets/training` |
| `GLIOMA_DATASET_ROOT` | `glioma_goals` 数据根（也接受 `DATASET_ROOT` / `DATASET_PATH`） | 各 goal 的 `config.yaml` |
| `GLIOMA_LABELS_DIR` | 官方 5 张标注表目录 | **平台实测**：表在**团队工作区**（如 `/2026aicompetition/workspace/dcs/goal1and2/Goal1and2/labels`），**不在**数据集挂载里；不设则自动按序找：`<工程>/labels` → `$WORKSPACE` 下 3 层内所有 `labels/` → 数据根/父/祖父 |
| `CACHE_DIR` | 预处理缓存（放私有存储） | `<workspace>/cache` |
| `WORKSPACE` | 平台工作区 | `/2026aicompetition/workspace` |
| `GLIOMA_CHECKPOINT_ROOT` | 权重导出根 | `$WORKSPACE/checkpoint` |
| `GLIOMA_FOLDS` | 共用折划分 | `glioma_track4/data/folds.json` |
| `PY` | python 解释器（容器里常只有 `python3`） | 自动探测：`$PY` → `python3` → `python`（`.sh` 已适配；直接跑 `.py` 用 `python3`） |

### 8.2 关键路径

```text
数据          /2026aicompetition/datasets/training/annotation
提交工程      /2026aicompetition/workspace/dcs/Glioma_recognition-main
训练工程 A    /2026aicompetition/workspace/dcs/glioma_track4
训练工程 B    /2026aicompetition/workspace/dcs/glioma_goals
权重          /2026aicompetition/workspace/checkpoint/<goal>/model.pt
答案          /2026aicompetition/workspace/answer/<evaluation_id>/
日志          /2026aicompetition/workspace/logs
```

### 8.3 `glioma_track4/scripts` 常用脚本

| 脚本 | 作用 |
|---|---|
| `00_setup_env.sh` / `00b_prepare_wheels.sh` | 环境与离线依赖准备 |
| `01_probe.sh [数据根]` | 数据探针 → `data/manifest.json` + 结构报告 |
| `02_build_dataset.sh` | 折划分与统计（含数据源合规闸门） |
| `03_train.sh 0 \| all 4 \| one` | 训练（单折 / 4 折并行）；支持 `FOLD=` `GPUS=` `PRETRAINED=` |
| `04_eval.sh` | 评估 |
| `05_serve.sh` / `06_platform_serve.sh` | 起本地服务 / 按平台协议起服务 |
| `07_export_weights.sh` | 把训练产物导出到平台**持久化目录** |
| `08_mock_competition.sh` | Mock 比赛：协议闭环自测 |
| `09_export_submission.sh` | 导出**提交权重**到 `checkpoint/<goal>/` |
| `13_build_cache.py --workers 8` | 预处理缓存 |
| `16_finalize.sh` | 收尾流水线（`FOLDS="0 1 2 3"`） |
| `17_reset_for_official.sh` | 归档本地验证产物，重置为官方数据流程 |
| `20_bridge_selftest.py` | 桥接集成自测 |
| `22_fault_injection.py` | 故障注入（22 项均需有合规答案） |
| `23_pre_submit_check.sh [--quick]` | **提交前检查清单（一条命令跑完全部验证）** |
| `24~26` | 评估划分 / 任务集成 / 插件完整性审计 |
| `29_locate_dataset_root.py` | 数据根定位（打印候选根与病例数） |
| `30_inspect_table.py` | 检查标注表结构 |
| `32_apply_abnormal_patch.py` | 核验并合并 `1_abnormal` 人工补丁（按 `Label` 拼路径做磁盘核对，命中率不升则拒绝写回） |
| `33_fix_abnormal_labels.sh` | **一条命令修好 `1_abnormal`**：自动定位原表/补丁/病例层 → 磁盘核对命中率 → 落地（自动备份）→ 重跑探针。`--dry-run` 只核验、`--no-probe` 跳过探针 |
| `99_smoke_test.py` | 端到端自检（合成数据，无需真实数据） |

> 解释器：容器里往往只有 `python3`。`01`~`05`、`33` 这几个 `.sh` 会自动探测（`$PY` → `python3` → `python`）；直接跑 `.py` 时把 `python` 换成 `python3`，或先 `export PY=python3`。

---

## 附录 A · 数据根下钻自检（可直接粘贴）

造一棵「真例在 `training/annotation/` 下」的树，验证下钻与护栏：

```bash
python - <<'PY'
import os, sys, tempfile
from pathlib import Path
import numpy as np, nibabel as nib

REPOS = "/2026aicompetition/workspace/dcs"      # ← 改成三工程所在目录
tmp = Path(tempfile.mkdtemp()); stage = tmp / "training"

def mk(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.int16), np.eye(4)), p)

mk(str(stage / "annotation" / "ACC001" / "S1" / "S1.nii.gz"))          # 真例（在 annotation/ 下）
mk(str(tmp / "datasets" / "training" / "A" / "S" / "S.nii.gz"))        # 多阶段父目录
mk(str(tmp / "datasets" / "evaluation_first" / "B" / "S" / "S.nii.gz"))

sys.path.insert(0, f"{REPOS}/glioma_track4")
from src.data import probe as P

print("① 填高一层自动下钻 :", P.resolve_case_root(str(stage)) == str(stage / "annotation"))
print("② 扫到真例         :", len(P.scan_real(str(stage))) == 1)
try:
    P.scan_real(str(tmp / "datasets")); print("③ 多阶段父目录护栏 : ✗ 失效")
except ValueError:
    print("③ 多阶段父目录护栏 : ✓ 仍报错")
PY
```

验证 `glioma_goals` 侧（同样自动下钻）：

```bash
cd glioma_goals && python -c "
import sys; sys.path.insert(0, '.')
from shared.data import discover_cases
cases = discover_cases('/tmp/plat/training')        # ← 换成数据根（可填高一层）
print('扫到病例:', len(cases))"
```

---

## 附录 B · 本地复现平台布局（用于离线验证）

```bash
D=/path/to/本地数据            # 目录形如 <检查号>/<模态>/*.nii.gz
P=/tmp/plat
rm -rf $P && mkdir -p $P/training/annotation
for e in "$D"/*/; do ln -s "$e" "$P/training/annotation/$(basename "$e")"; done
rm -f $P/training/annotation/annotation          # 去掉自身造成的嵌套

# 数据根填**上一层**，验证自动下钻
export DATASET_ROOT=$P/training
bash scripts/01_probe.sh
cd ../glioma_goals && python smoke_all_goals.py --datasets-only --data $P/training --limit 8
```

> 用符号链接即可：两条发现逻辑都基于 `os.listdir` / `Path.iterdir`，会正常跟随链接。
> 若输出 0 例，改用 `cp -r` 复制一份再试。

