#!/usr/bin/env python
"""定位"真正的数据根"（容器内排查 `/2026aicompetition/datasets` 用）。

**为什么需要它**：平台把数据集挂在 `/2026aicompetition/` 下，但

* 官方训练集在文档里叫 ``public_dataset_{赛道}``，实际可能叫 ``datasets/``；
* 实例创建时若只勾了标注那份存储，某个阶段目录下会**只有 annotation/**，
  影像根本不在容器里；
* 数据根必须精确到"含检查号目录的那一层"，停在上层会得到 5 个以阶段名
  命名的假病例（或直接触发 ``assert_case_root`` 护栏）。

人眼 ``ls`` 很容易看错层级，所以这里让**工程自己的发现逻辑**给出结论：
能扫出非零病例数的那个路径才是数据根。

用法::

    python scripts/29_locate_dataset_root.py                      # 默认扫 /2026aicompetition
    python scripts/29_locate_dataset_root.py --base /mnt/data/...  # 本地演练
    python scripts/29_locate_dataset_root.py --max-depth 6         # 数据藏得比较深时

输出三段：① 顶层挂载点 → ② 哪里有 NIfTI → ③ 各候选根的病例数（带结论）。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

#: 扫描时剪掉的目录名（自家缓存/权重目录，里面即使有 NIfTI 也不是数据集）
_PRUNE = {"cache", "checkpoints", "__pycache__", ".git", "runs", "logs"}

#: 平台阶段目录名（与推理侧 Loader、训练侧护栏共用同一份清单）
_PHASES = ("training", "evaluation_first", "evaluation_second",
           "evaluation_finals", "verification")


def _default_goals() -> Path:
    """定位 ``glioma_goals``（用于复用它的 ``discover_cases``）。"""
    env = os.environ.get("GLIOMA_GOALS_ROOT")
    if env and (Path(env) / "shared" / "data.py").is_file():
        return Path(env)
    here = Path(__file__).resolve().parent
    cands = [here.parent.parent / "glioma_goals",
             here.parent.parent / "workspace" / "dcs" / "glioma_goals",
             Path("/2026aicompetition/workspace/dcs/glioma_goals")]
    for c in cands:
        if (c / "shared" / "data.py").is_file():
            return c
    return cands[0]


#: 序列目录的常见命名（`flair_0000` / `t1c` / `T2` …）
_MODALITY_STEM = {"flair", "t1", "t1c", "t1ce", "t1w", "t2", "t2w", "dwi",
                  "adc", "swi", "bold", "perf", "image", "img", "volume"}

#: 大赛检查号的形态：**32 位十六进制**（实测 ``0050d79429cf4d86907dc8c4a34cbf04``）。
_ACC_RE = re.compile(r"^[0-9a-f]{32}$")

#: 平台挂载点下**一定属于本赛道**的顶层目录名（§① 只列这些 + 大赛数据树）。
_PLATFORM_TOP = {"datasets", "workspace"}


def _has_official_case(root: Path) -> bool:
    """该目录是不是大赛数据布局：一级子目录里**有 32 位十六进制检查号**。

    平台契约是 ``<数据根>/<检查号>/…``，检查号必然是 32 位十六进制 —— 这是
    "只读大赛数据"的**唯一判据**。不满足的目录（别的数据集、随手堆着 nii 的目录）
    **连探测都不做、名字都不打印**：读了会污染数据，报出来会把排查方向带偏。
    """
    try:
        return any(k.is_dir() and _ACC_RE.match(k.name) for k in root.iterdir())
    except OSError:                                             # noqa: BLE001
        return False


def _looks_like_root(p: Path) -> bool:
    """区分**数据根**与**检查目录**（两者都能被 ``discover_cases`` 扫出非零数）。

    这是本脚本最容易误导人的地方：``discover_cases`` 只认"一级子目录 = 检查号"，
    所以把 ``<数据根>/<检查号>`` 当成根传进去，它会把**序列目录**（``flair_0000``）
    当成检查号，照样返回一个非零数字——看起来"可用"，实则少了一层。

    两者结构相同，只能靠命名区分：检查号的子目录是模态名，数据根的子目录是病例号。
    """
    kids = [k for k in p.iterdir() if k.is_dir() and k.name.lower() not in _PRUNE]
    if not kids:
        return False
    modality_like = 0
    for k in kids:
        stem = re.sub(r"[_\-\s]*\d+$", "", k.name.casefold()).strip("_- ")
        if stem in _MODALITY_STEM:
            modality_like += 1
    return modality_like < len(kids)


def _simple_count(root: Path) -> int:
    """内置的病例计数（``discover_cases`` 不可用时的兜底）。

    规则与工程一致：一级子目录 = 检查号（**只认 32 位十六进制**），其下递归找
    NIfTI，跳过标注类目录与掩膜文件。只做粗略判定，够用来比较"哪个候选有数据"。
    """
    skip_dirs = {"annotation", "cache", "runs", "folds", "labels"}
    mask_hints = ("mask", "seg", "label", "roi", "掩码", "标注",
                  "瘤体", "水肿", "异常", "核心", "病灶", "肿瘤区")
    n = 0
    for acc in sorted(p for p in root.iterdir() if p.is_dir()):
        if acc.name.lower() in skip_dirs or not _ACC_RE.match(acc.name):
            continue
        for f in acc.rglob("*"):
            if not f.is_file() or not f.name.lower().endswith((".nii", ".nii.gz")):
                continue
            if any(h in f.name.lower() for h in mask_hints):
                continue
            n += 1
            break
    return n


def _in_competition_tree(base: Path, d: Path) -> bool:
    """该目录属于**大赛数据树**吗（决定 §①/§② 要不要把它列出来）。

    判据（任一成立即可）：
      · 路径里出现 32 位十六进制检查号目录（平台的病例目录名）；
      · 该目录、父目录或祖父目录里直接放着 ``SeriesType.xlsx``（大赛的数据信息表）。

    不满足的目录**一律不列**：§①/§② 是回答"影像挂上了没有 / 数据根在哪一层"，
    不是列全机 NIfTI —— 把无关目录列出来只会把排查方向带偏（硬口径，见
    ``26_audit_plugin_completeness.py`` 的守门断言）。
    """
    for anc in (d, d.parent, d.parent.parent):
        if (anc / "SeriesType.xlsx").is_file():
            return True
    try:
        parts = d.relative_to(base).parts
    except ValueError:
        parts = d.parts
    return any(_ACC_RE.match(x) for x in parts)


def _scan_nifti(base: Path, max_depth: int) -> dict[str, int]:
    hits: dict[str, int] = {}
    for root, dirs, files in os.walk(base):
        if len(Path(root).relative_to(base).parts) >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in _PRUNE]
        n = sum(1 for f in files if f.endswith((".nii", ".nii.gz")))
        if n and _in_competition_tree(base, Path(root)):
            hits[root] = n
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="定位真正的数据根")
    ap.add_argument("--base", default="/2026aicompetition",
                    help="挂载点根（默认 /2026aicompetition）")
    ap.add_argument("--goals", default=None, help="glioma_goals 路径（默认自动探测）")
    ap.add_argument("--max-depth", type=int, default=5, help="扫描深度上限")
    a = ap.parse_args()

    base = Path(a.base).expanduser()
    print("=" * 72)
    print(f"① 顶层挂载点：{base}")
    print("=" * 72)
    if not base.is_dir():
        print(f"  {base} 不存在 —— 当前不在比赛容器里，或挂载点不同名。")
        print("  用 --base 指向实际挂载点后重跑。")
        return 2
    for p in sorted(base.iterdir()):
        # 只列平台目录（datasets/workspace）与**大赛数据树**里的条目；其余**不列**
        # （硬口径：与大赛无关的数据、无关的输出一个都不要有）。
        if p.name not in _PLATFORM_TOP and not _in_competition_tree(base, p):
            continue
        try:
            n = sum(1 for _ in p.iterdir()) if p.is_dir() else 0
            print(f"  {'dir ' if p.is_dir() else 'file'} {p}   子项={n}")
        except PermissionError:
            print(f"  dir  {p}   (无权限)")

    print()
    print("=" * 72)
    print(f"② 哪里有 NIfTI（深度≤{a.max_depth}，按数量排序；只列大赛数据树）")
    print("=" * 72)
    hits = _scan_nifti(base, a.max_depth)
    for d, n in sorted(hits.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {n:>7} 个 NIfTI  {d}")
    if not hits:
        print("  ⚠️ 未在**大赛数据树**里找到 .nii/.nii.gz —— 影像那份存储很可能没挂上。")
        print("     这不是代码问题：回容器实例的「存储与数据服务」勾选训练/评测影像数据集，")
        print("     或重建实例后再跑一次本脚本。")

    print()
    print("=" * 72)
    print("③ 候选数据根的病例数（用工程自己的发现逻辑）")
    print("=" * 72)
    goals = Path(a.goals).expanduser() if a.goals else _default_goals()
    discover = None
    if (goals / "shared" / "data.py").is_file():
        sys.path.insert(0, str(goals))
        cwd = os.getcwd()
        try:
            os.chdir(goals)                       # shared 内部用相对导入/相对路径
            from shared.data import discover_cases          # noqa: PLC0415
            discover = discover_cases
            print(f"  （使用 {goals}/shared/data.py 的 discover_cases）")
        except Exception as exc:                            # noqa: BLE001
            print(f"  （导入 discover_cases 失败：{type(exc).__name__}: {exc}）")
        finally:
            os.chdir(cwd)
    else:
        print(f"  （找不到 {goals}/shared/data.py，改用内置计数）")

    cands = [base / "datasets", base / "datasets" / "training",
             base / "datasets" / "training" / "annotation"]
    # 从 NIfTI 命中位置**上推**候选根：只取离挂载点较近的祖先（rel 深度 ≤3），
    # 否则会把 ``<sequence>_0000`` 这类叶目录也当候选，刷屏且没有信息量。
    for d in hits:
        p = Path(d)
        for anc in p.parents:
            try:
                rel = len(anc.relative_to(base).parts)
            except ValueError:
                break
            if rel == 0:
                break
            if rel <= 3:
                cands.append(anc)
    cands += [base / f"public_dataset_{t}" for t in ("", "四", "4")]

    seen: set[str] = set()
    usable: list[tuple[str, int]] = []          # 大赛数据根（0 例的也留，便于看"挂没挂上"）
    other: list[tuple[int, str]] = []           # (排序键, 已格式化文本)
    for c in cands:
        key = str(c)
        if key in seen or not c.is_dir():
            continue
        seen.add(key)
        # ★ 只探测**大赛数据布局**的目录（一级子目录里有 32 位十六进制检查号）。
        #   其余目录**连探测都不做、连名字都不打印** —— 这是"不读非大赛数据、
        #   不输出非大赛信息"的执行点（此前会把机器上别的数据集一并扫出来报）。
        if not _has_official_case(c):
            continue
        try:
            # 探测模式：对**多个候选根**轮流试，紧凑输出（由本段汇总）。
            if discover:
                try:
                    _cs = discover(c, diagnose=False)
                except TypeError:                  # 旧版签名（无 diagnose 参数）
                    _cs = discover(c)
                n = len(_cs)
            else:
                n = int(_simple_count(c))
        except Exception as exc:                            # noqa: BLE001
            other.append((2, f"  {key:<56} 拦截: {type(exc).__name__}"
                             f"（父目录/无效根，预期行为）"))
            continue
        if n and not _looks_like_root(c):
            other.append((1, f"  {key:<56} {n}（子目录全是模态名 →"
                             f"这是**检查目录**，不是数据根）"))
            continue
        usable.append((key, n))

    # 有数据的排前面：结论一眼可见，不必在噪声里找
    winners = [(k, n) for k, n in usable if n]
    for key, n in sorted(usable, key=lambda kv: -kv[1]):
        print(f"  {key:<56} -> {n} 例" + ("  ← 可用" if n else ""))
    for _, text in sorted(other):
        print(text)

    print()
    print("=" * 72)
    if winners:
        best = max(winners, key=lambda kv: kv[1])
        print(f"结论：数据根 = {best[0]}（{best[1]} 例）")
        print(f"      export DATASET_ROOT={best[0]}")
        if "/training" not in best[0] and "training" not in best[0].split("/"):
            print("      注意：这不是 training/ 那层 —— 若含 evaluation_* 阶段名，")
            print("            说明你扫到的是评测集；训练要用含训练影像的那个根。")
    else:
        print("结论：候选里**没有大赛数据布局的根**（一级子目录须含 32 位十六进制")
        print("      检查号，如 0050d79429cf4d86907dc8c4a34cbf04）。影像可能未挂载。")
        print("      请直接指定根并核对挂载：export DATASET_ROOT=…/datasets/training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
