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
结论：数据根 = /2026aicompetition/datasets/training（204 例）
      export DATASET_ROOT=/2026aicompetition/datasets/training
```

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
