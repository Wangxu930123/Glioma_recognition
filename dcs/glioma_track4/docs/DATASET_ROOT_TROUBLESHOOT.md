# 数据根排查（容器内 `/2026aicompetition`）

> 适用症状：
> - `/2026aicompetition/datasets/training` 下**只有 `annotation/`**，看不到影像；
> - 跑探针/发现病例时报
>   `ValueError: 数据根 /2026aicompetition/datasets 指向数据集父目录，其下是平台阶段目录 [...]`；
> - 病例数正常、却能训练出 0 个样本，报 `无任何可用序列`
>   （→ 直接看 [「病例数正常、却报无任何可用序列」](#病例数正常却报无任何可用序列)）；
> - 不确定 `DATASET_ROOT` 到底该填哪一层。

## 一条命令解决

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
python scripts/29_locate_dataset_root.py
```

它会打印三段：**① 顶层挂载点 → ② 哪里有 NIfTI → ③ 各候选根的病例数**，
并在最后直接给出 `export DATASET_ROOT=...` 那一行。

数据藏得比较深、或挂载点不叫 `datasets` 时：

```bash
python scripts/29_locate_dataset_root.py --base /2026aicompetition --max-depth 6
python scripts/29_locate_dataset_root.py --base /2026aicompetition --goals ../glioma_goals
```

## 输出怎么读

| 输出 | 含义 | 动作 |
|---|---|---|
| ② 列出 `N 个 NIfTI  <目录>` | 影像真实位置 | 数据根取**含检查号目录的那一层**（通常是它的上一层） |
| ② 提示"未找到任何 `.nii/.nii.gz`" | 影像没挂到实例上 | 回容器实例的「存储与数据服务」勾选影像数据集，或重建实例——**不是代码问题** |
| ③ 某行 `-> N 例  ← 可用` | 可直接用的数据根 | `export DATASET_ROOT=<该行路径>` |
| ③ 某行 `拦截: ValueError: ...父目录...` | 该路径是父目录 | 预期行为，见下节；看**第一个非零行**即可 |

结论行示例：

```text
结论：数据根 = /2026aicompetition/datasets/training/annotation（3255 例）
      export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
```

> 填 `.../training` 也一样能用：两个工程都会自动下钻到 `annotation/` 并打印告警
> （见 `CLOUD_DESKTOP_RUNBOOK.md` §3.2）。只有多阶段的父目录 `/2026aicompetition/datasets`
> 会被 `ValueError` 拦住。

## 为什么 `training/` 下会只有 `annotation/`

`/2026aicompetition/` 不是普通目录，而是平台按**你创建实例时勾选的存储**逐份挂载的。
按《训推平台使用指南》：

- 公共存储（只读）里同时有**公共模型**和**训练集**，训练集名为 `public_dataset_{赛道}`；
- 创建实例时要在「存储与数据服务」里**显式勾选**要挂载的目录；
- 未勾选的存储，在容器里就是不存在的。

因此"只剩 `annotation`"最常见的三种原因：

1. **只挂上了标注那份存储**，影像那份没勾（最常见，② 段会直接暴露出"全盘没有 NIfTI"）；
2. 影像挂在 **`public_dataset_{赛道}`** 而不是 `datasets/`（命名差异，同时看 ① 段的顶层列表）；
3. 影像是**布局 B**：直接躺在 `datasets/` 根下，`training/` 只放了标注
   （则数据根取 `datasets` 本身，③ 段那一行会是非零）。

## 为什么父目录会直接报错

```python
discover_cases(Path("/2026aicompetition/datasets"))     # ValueError
```

数据根必须精确到"含检查号目录的那一层"。停在父目录时，`evaluation_first`
这类**阶段名会被当成检查号**：训练照常启动、损失照常下降，但输入是几份数据混在
一起的像素，金标准一张也对不上，直到提交才会发现全错。所以两条训练路径
（`glioma_goals/shared/data.py: assert_case_root()`、
`glioma_track4/src/data/probe.py: assert_case_root()`）都在扫描前**直接失败**，
并在报错里给出应该填的路径。

只想临时扫一下父目录看内容时，把调用包进 `try/except` 即可（第 29 号脚本就是这么做的）；
**不要**为此删掉护栏。

## 备用：不依赖脚本的等价片段

脚本路径找不到时，把下面这段存成 `locate.py` 放到 `glioma_goals/` 下再 `python locate.py`：

```python
import os, sys
from pathlib import Path

BASE = Path("/2026aicompetition")
GOALS = Path("/2026aicompetition/workspace/dcs/glioma_goals")

print("=== ① 顶层挂载点 ===")
for p in sorted(BASE.iterdir()):
    try:
        n = sum(1 for _ in p.iterdir()) if p.is_dir() else 0
        print(f"  {'dir ' if p.is_dir() else 'file'} {p}   子项={n}")
    except PermissionError:
        print(f"  dir  {p}   (无权限)")

print("\n=== ② 哪里有 NIfTI（深度≤5）===")
PRUNE = {"cache", "checkpoints", "__pycache__", ".git", "runs", "logs"}
hits = {}
for root, dirs, files in os.walk(BASE):
    if len(Path(root).relative_to(BASE).parts) >= 5:
        dirs[:] = []
    dirs[:] = [d for d in dirs if d not in PRUNE]
    n = sum(1 for f in files if f.endswith((".nii", ".nii.gz")))
    if n:
        hits[root] = n
for d, n in sorted(hits.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {n:>6} 个 NIfTI  {d}")
if not hits:
    print("  （没找到任何 .nii/.nii.gz —— 影像那份存储没挂上）")

print("\n=== ③ 候选数据根的病例数 ===")
os.chdir(GOALS); sys.path.insert(0, str(GOALS))
from shared.data import discover_cases
cands = ["/2026aicompetition/datasets",
         "/2026aicompetition/datasets/training",
         "/2026aicompetition/datasets/training/annotation"] + sorted(hits)[:8]
for c in dict.fromkeys(cands):
    p = Path(c)
    if not p.is_dir():
        print(f"  {c:<56} (不存在)"); continue
    try:
        cs = discover_cases(p)
    except Exception as e:
        print(f"  {c:<56} 拦截: {type(e).__name__}: {str(e)[:70]}"); continue
    print(f"  {c:<56} -> {len(cs)} 例  {[x['accession'] for x in cs[:3]]}")
```

## 病例数正常、却报「无任何可用序列」

```text
数据根: /2026aicompetition/datasets/training/annotation
[data] 未找到可用折划分，按 val_ratio=0.2 自行划分（train=2604 val=651）
ValueError: Caught ValueError in DataLoader worker process
  ... build_volume ... ValueError: study '3255123456' 无任何可用序列
```

**病例数扫得出来（示例里 3255 例），但每个病例都挑不出模态** —— 这是训练侧
"模态识别"的问题，不是数据损坏、也不是路径错。

原因：序列的模态**只能从数据信息表取**，而数据的序列目录名是 **DICOM UID**
（`2.25.25750572698...`）、病例目录名是 32 位哈希，任何"按名字猜关键词"都命中不了，
于是 `pick_series` 返回空 → 报错。

**数据路径与数据信息路径**（以训练集为例，验证集 `verification/` 同构）：

```text
★ 影像数据：  /2026aicompetition/datasets/training/annotation/<32位检查号>/<2.25.* 序列UID>/<序列UID>.nii.gz
★ 数据信息：  /2026aicompetition/datasets/training/annotation/SeriesType.xlsx   （与病例目录同层）
              列 AccessionNumber + SeriesUid + SeriesType ∈ {T1, T1CE（增强）, T2-Flair, T2WI, 其他}
              （评测集的这张表在正式测试时随测试数据一起下发）
```

取值优先级：

```text
SeriesType.xlsx（数据集自带，权威）→ 3_serieslabel.xlsx（工作区那份，兼容兜底）→ 同名 .json sidecar → 目录名（本地模拟集仍照旧）
```

| 名字 | 在哪 | 列 / 取值 |
|---|---|---|
| `SeriesType.xlsx` | **赛道四数据集里**：`<阶段>/annotation/SeriesType.xlsx`（与病例目录**同层**） | `AccessionNumber`, `SeriesUid`, `SeriesType` ∈ {`T1`, `T1CE（增强）`, `T2-Flair`, `T2WI`, **`其他`**} |
| `3_serieslabel.xlsx` | 团队工作区 `labels/`（**不是本赛道数据集的内容**，属另一个目标；仅兜底） | `AccessionNumber`, `SeriesUid`, `SeriesLabel` ∈ {`T1CE`, `T2`, `FLAIR`} |

> `1_abnormal.xlsx` / `2_duplicate.xlsx` / `3_serieslabel.xlsx` / `4_masklabel.xlsx` /
> `5_characteristics.xlsx` 这 5 张表**与赛道四数据集无关**，不必为它们去配置路径。
>
> **两个名字都找，但顺序不能反**：`SeriesType.xlsx` 先、且**只补缺不覆盖**。
> 反过来会让工作区那份（取值更粗，只写 `T2`）静默覆盖数据集的 `T2WI` / `T2-Flair` ——
> 表现是"模态看着都认出来了、通道里却是错的对比度"，比直接报错难查得多。
> 现在统一按候选目录搜（`$GLIOMA_LABELS_DIR` → `<工程>/labels` → `$WORKSPACE` 下 3 层 →
> 数据根/父/祖父 → 像标注容器的子目录），所以把数据根指成 `.../training`、
> `annotation/` **或**某一病例目录都能命中 —— 现场表现就是"表就在磁盘上，报错却说没找到"。
>
> `SeriesType` 里的 **`其他`** 是权威结论（该序列不是 T1/T1CE/T2-Flair/T2WI 中的任何一个），
> 探针会直接排除、**不交给体素判别模型猜**：模型只认识 4 类，容易把 DWI/ADC 判成
> `t2`（还没法验证），往通道里灌错对比度比留空更有害。报告里
> `cases_with_declared_other_series` 就是它的计数，**非 0 属正常，不代表缺表**。

对应的代码：

| 工程 | 位置 | 作用 |
|---|---|---|
| `glioma_goals` | `shared/data.py: discover_cases / read_series_types` | 训练取数（`train.py` 走这条） |
| `glioma_track4` | `src/data/probe.py: scan_real / _collect_nifti`、`src/data/labels.py: read_series_types` | 探针、缓存、独立推理 |

### 先跑这一条命令

```bash
cd <你平时跑 01_probe.sh 的 glioma_track4 目录>   # 容器里通常是 /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/01_probe.sh
```

只看**四个数**（探针报告 + 末尾告警里都有）：

| 报告字段 | 期望 | 含义 |
|---|---|---|
| `series_type_rows` | **> 0** | 类型表读到 N 条 `(检查号, 序列号) → 模态` |
| `modality_counts` | 出现 `t1c` / `flair` / `t2` | 模态认出来了 |
| `unknown_series_total` | **0** | 没有需要兜底模型去猜的序列 |
| `cases_with_declared_other_series` | 任意（**非 0 正常**） | 被类型表标为 `其他` 的病例数；这些序列**已排除**、不进模型，不表示缺表 |

**接了表必须重跑探针**（训练读的是 `data/manifest.json`，不重跑不生效），之后
`bash scripts/02_build_dataset.sh`。

### 表在哪：默认会自动找（零配置）

探针按这个顺序定位**序列类型表**，**命中即止**（`SeriesType.xlsx` 与 `3_serieslabel.xlsx`
走**同一批**候选目录，两个名字分别找）：

```text
显式 --labels / $GLIOMA_LABELS_DIR  →  <工程>/labels  →  $WORKSPACE 下 3 层内所有 labels/  →  数据根/父/祖父
```

最后两项就是为**数据集里的** `SeriesType.xlsx` 准备的：它与病例目录同层（都在
`annotation/` 里），所以数据根填 `.../training`、`.../training/annotation` 或
`annotation/<某病例目录>` 都能命中。

团队工作区里那份 `3_serieslabel.xlsx`（`/2026aicompetition/workspace/dcs/goal1and2/Goal1and2/labels/`，
**与数据集无关**）**通常不做任何操作就能被找到**（搜索有界：深度 ≤ 3、只认名为 `labels`
的目录、跳过 `cache`/`logs` 等），只在数据集那份读不到时才起作用。工作区在别处或层级更深时，
二选一显式接上：

```bash
# 方式①（推荐，一次到位）：软链到工程目录，之后所有脚本都认
ln -s /2026aicompetition/workspace/dcs/goal1and2/Goal1and2/labels labels

# 方式②：每个会话 export 一次
export GLIOMA_LABELS_DIR=/2026aicompetition/workspace/dcs/goal1and2/Goal1and2/labels
```

> 表里的"检查号"列与磁盘病例目录名**对不上也没关系**：只要 `SeriesUid` 与影像
> 同源（官方数据必然如此），会自动按 **UID 单键回退**命中，不必改表。

### 还是 0：三条诊断命令

```bash
# ① 表到底在哪（两个命名都搜：平台那份在数据里，团队那份在工作区）
find /2026aicompetition/datasets /2026aicompetition/workspace -maxdepth 5 \
     \( -name "SeriesType.xlsx" -o -name "3_serieslabel.xlsx" \) 2>/dev/null

# ② 表与影像是不是同一批（看 SeriesUid ∩ 磁盘序列名 是否 > 0）
python3 - <<'PY'
import os, pandas as pd
ACC  = sorted(os.listdir("/2026aicompetition/datasets/training/annotation"))[0]
ROOT = f"/2026aicompetition/datasets/training/annotation/{ACC}"
# 平台下发的那份：与病例目录同层；若换成团队那份，指向 labels/3_serieslabel.xlsx
LAB  = "/2026aicompetition/datasets/training/annotation/SeriesType.xlsx"
df   = pd.read_excel(LAB, dtype=str)
uid  = next(c for c in df.columns if c.lower() in ("seriesuid", "series_uid", "序列号"))
seqs = {s for s in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{s}")}
print("病例:", ACC, "| 磁盘序列:", len(seqs), "| 表内 UID ∩ 磁盘 =", len(set(df[uid].astype(str)) & seqs))
print("表的列名:", list(df.columns))
PY

# ③ 一个病例的完整结构（贴出来就能定位是哪一层的问题）
ls -la /2026aicompetition/datasets/training/annotation/<某个检查号>/
ls -la /2026aicompetition/datasets/training/annotation/<某个检查号>/<某个序列目录>/
```

> 报错里若是"**共 2 条序列**"而正常检查应有 3~4 路（T1C/T2/FLAIR[/T1]）：
> 先用 ③ 区分是"这个检查确实只有 2 路"还是"数据根指深/浅了一层"。

## 连类型表都没有：用体素统计模型兜底判模态

**这是评测期的默认情形**：训练/验证集的数据里都有 ``SeriesType.xlsx``，
**评测集不给任何标注**（`SeriesType.xlsx` 也在正式测试时才随测试数据一起下发），
序列目录名是 DICOM UID。表还没到手、或表里没有这个检查号时若一个模态都认不出来：

* 训练侧：`无任何可用序列` 直接崩；
* 推理侧更隐蔽：`inference/pipeline.py` 要先知道"哪个序列是 T1C"才能把掩膜
  **写回它的空间**，认不出就写不回去 —— 提交上去的掩膜空间是错的。

官方为这种情况在推理主链里放了一个 `sequence` 任务（3 分类）专门判模态。
本工程的对等实现是**体素统计特征 + 逻辑回归**（无 GPU、无额外依赖，模型 ~2KB）：

```bash
# 用官方训练集训练（标签来自数据信息 <阶段>/annotation/SeriesType.xlsx）—— 生产推荐路径
python3 scripts/31_train_modality_model.py --root $DATASET_ROOT

# 只看精度不写模型（5 折 + 混淆矩阵 + 特征权重）
python3 scripts/31_train_modality_model.py --root $DATASET_ROOT --dry-run

# 本地模拟集（标签来自目录名 flair_0000 / t1c_0000 …）
python3 scripts/31_train_modality_model.py --root /path/to/track4_sim
```

> 训练侧那条 `无任何可用序列` 报错**已经自带自检**：会打印"标注表找到了没 / 体素模型
> 在不在"以及上面两条命令，照着做即可，不用回来翻文档。

产出 `data/modality_model.json`，由 `src/data/dataset.py: pick_series` 在
**按名字挑不出通道时**自动调用（懒加载：常规路径零开销）。判别依据是物理量，
不是黑盒：T2 的脑脊液亮（`bright_frac`）、FLAIR 的脑脊液被抑制（`dark_frac`）、
T1CE 的增强灶（高强度尾部）—— 权重可直接打印检视。

**两个必须知道的行为**：

1. **判错比判不出更糟**：置信度 < 0.5 一律弃用（留空通道 = 明确的"缺失"），
   把 FLAIR 当成 T1C 会把水肿送进"增强核心区"通道；
2. **多路之间做全局贪心一对一指派**，不是逐路先到先得 —— 否则 argmax 撞车时
   会白丢一路（详见 `dataset.classify_unknown` 的注释）。

### 精度与跨域风险（务必如实看待）

| 训练数据 | 5 折准确率 | 对**评测集**的可靠性 |
|---|---|---|
| 本地模拟集（BraTS 派生，750 例） | **0.960** | **未知** —— 与官方数据不同厂商/协议，存在域差 |
| 官方训练集（`--root $DATASET_ROOT`） | 现场测得 | 同分布，**这才是生产路径** |

模拟集训出的模型只保证**管线是通的**（`scripts/25_verify_tasks_integration.py`
第 ⑱ 段用**训练未见过**的病例做行为级断言）。真正上评测前，请用官方训练集重训一次
—— 标签现成，CPU 几分钟，没有任何理由跳过。

## 定根之后

```bash
export DATASET_ROOT=<结论给出的路径>
export CACHE_DIR=/2026aicompetition/workspace/cache        # 缓存放私有存储（容器删了不丢）

# 训练：先探针（确认病例数、模态、掩码角色、金标准命中数都正常）
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/01_probe.sh
# 六个 Goal 的能训练自检（首轮建议用 --datasets-only，秒级）
cd ../glioma_goals && python smoke_all_goals.py --datasets-only --limit 8
```

推理侧的数据根是平台通过 `/call` 的 `input.dataset_path` 传进来的，**不需要配置**；
它指向哪个阶段目录就处理哪个，误传父目录同样会被 Loader 当场拦下。
