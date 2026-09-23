# 数据根排查（容器内 `/2026aicompetition`）

> 适用症状：
> - `/2026aicompetition/datasets/training` 下**只有 `annotation/`**，看不到影像；
> - 跑探针/发现病例时报
>   `ValueError: 数据根 /2026aicompetition/datasets 指向数据集父目录，其下是平台阶段目录 [...]`；
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

原因：序列的模态原本**只能从目录名猜**（`shared/data.py: _AsSeries.modality = s["uid"]`）。
本地模拟集的目录名是 `flair_0000` / `t1c_0000`，所以一直没暴露；官方数据的
序列目录名是 **DICOM UID**（`1.2.826.0.1.3680043.2.1125.1.1001`），任何关键词
都命中不了，于是 `pick_series` 返回空 → 报错。

正确来源是官方数据根下的 **`SeriesType.xlsx`**（`AccessionNumber + SeriesUid →
SeriesType`），提交工程一直用它的 `data/metadata.py`，训练侧此前没实现。
现已补齐两条训练路径，取值优先级为：

```text
SeriesType.xlsx  →  同名 .json sidecar  →  目录名（模拟集仍照旧）
```

对应的代码：

| 工程 | 位置 | 作用 |
|---|---|---|
| `glioma_goals` | `shared/data.py: discover_cases / read_series_types` | 训练取数（`train.py` 走这条） |
| `glioma_track4` | `src/data/probe.py: scan_real / _collect_nifti`、`src/data/labels.py: read_series_types` | 探针、缓存、独立推理 |

### 怎么确认它生效

```bash
python -m src.data.probe --root <数据根> --out /tmp/probe.json | grep series_type
# 期望：[probe] 已读取 SeriesType.xlsx：N 条序列类型映射
#       报告里 "series_type_rows": N（不是 0）
```

`series_type_rows: 0` 且模态全是 `other` → 类型表**不在数据根那一层**。
它在哪一层，数据根就该填哪一层（`SeriesType.xlsx` 与 `<检查号>/` 目录同级）：

```bash
find /2026aicompetition/datasets -maxdepth 4 -name "SeriesType.xlsx"
```

找不到任何 `SeriesType.xlsx` 时，请把**一个病例目录的完整结构**贴出来：

```bash
ls -la /2026aicompetition/datasets/training/annotation/<某个检查号>/
ls -la /2026aicompetition/datasets/training/annotation/<某个检查号>/<某个序列目录>/
```

有了这两条，就能确定模态还能从哪里取（目录名约定 / sidecar 字段名 / 其它映射表），
不必再靠猜。

## 连类型表都没有：用体素统计模型兜底判模态

**这是评测期的默认情形**：官方训练集给了 `labels/3_serieslabel.xlsx`，
**评测集不给任何标注**，序列目录名是 DICOM UID。此时若一个模态都认不出来：

* 训练侧：`无任何可用序列` 直接崩；
* 推理侧更隐蔽：`inference/pipeline.py` 要先知道"哪个序列是 T1C"才能把掩膜
  **写回它的空间**，认不出就写不回去 —— 提交上去的掩膜空间是错的。

官方为这种情况在推理主链里放了一个 `sequence` 任务（3 分类）专门判模态。
本工程的对等实现是**体素统计特征 + 逻辑回归**（无 GPU、无额外依赖，模型 ~2KB）：

```bash
# 用官方训练集训练（标签来自 labels/3_serieslabel.xlsx）—— 生产推荐路径
python scripts/31_train_modality_model.py --root $DATASET_ROOT

# 只看精度不写模型（5 折 + 混淆矩阵 + 特征权重）
python scripts/31_train_modality_model.py --root $DATASET_ROOT --dry-run

# 本地模拟集（标签来自目录名 flair_0000 / t1c_0000 …）
python scripts/31_train_modality_model.py --root /path/to/track4_sim
```

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
