#!/usr/bin/env python
"""训练模态判别模型（评测期没有 ``3_serieslabel.xlsx`` 时的兜底）。

**为什么需要**：官方训练集给了序列标注，评测集**不给**；而评测集的序列目录名是
DICOM UID，靠关键词一个模态也挑不出来 —— 官方为此在推理主链里放了一个
``sequence`` 任务（3 分类）。本脚本用体素统计特征 + 逻辑回归训练一个轻量替代：
CPU 秒级训练、单例秒级预测、模型只有几百个参数。

用法::

    # 用官方训练集（标签来自 labels/3_serieslabel.xlsx）
    python scripts/31_train_modality_model.py --root $DATASET_ROOT

    # 用本地模拟集（标签来自目录名 flair_0000 / t1c_0000 …）
    python scripts/31_train_modality_model.py --root /path/to/sim --stride 3

    # 只报告精度、不写模型
    python scripts/31_train_modality_model.py --root $DATASET_ROOT --dry-run

输出：5 折交叉验证准确率 + 混淆矩阵 + 各特征权重（便于判断是否学到了物理依据），
模型默认写到 ``data/modality_model.json``。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.modality_model import (DEFAULT_MODEL_PATH, FEATURE_NAMES,   # noqa: E402
                                     OFFICIAL_CLASSES, ModalityModel,
                                     features_from_file)

#: 我们内部的模态键 → 官方 ``3_serieslabel.xlsx`` 的取值
MOD_TO_OFFICIAL = {"t1c": "T1CE", "t1ce": "T1CE", "t2": "T2", "flair": "FLAIR"}

#: 官方只有这三个模态（``schema.MODALITIES``）；T1 不在其中，训练时跳过
DEFAULT_CLASSES = OFFICIAL_CLASSES


def _label_map(classes: tuple[str, ...]) -> dict[str, str]:
    up = {c.upper() for c in classes}
    return {k: v for k, v in MOD_TO_OFFICIAL.items() if v.upper() in up}


def _features(args: tuple[str, int]) -> tuple[np.ndarray, str] | None:
    path, stride = args
    try:
        return features_from_file(path, stride=stride), path
    except Exception:                                             # noqa: BLE001
        return None


def collect(root: str, classes: tuple[str, ...], max_per_class: int) -> list[tuple[str, str]]:
    """收集 ``(nifti 路径, 官方模态)``。

    标签来源与训练/探针完全一致：官方 ``3_serieslabel.xlsx`` 优先，其次是
    目录名（本地模拟集）。**两条路径都走 ``scan_real``**，避免这里另起一套
    解析规则而与训练数据不一致。
    """
    from src.data.probe import scan_real

    keep = _label_map(classes)
    per_class: dict[str, list[str]] = defaultdict(list)
    for case in scan_real(root):
        for mod, meta in (case.get("images") or {}).items():
            label = keep.get(str(mod).lower())
            if label and len(per_class[label]) < max_per_class:
                per_class[label].append(str(meta["path"]))
    samples = [(p, lab) for lab, paths in per_class.items() for p in paths]
    return samples


def _confusion(y_true: list[str], y_pred: list[str], classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="训练模态判别模型（评测期兜底）")
    ap.add_argument("--root", default=os.environ.get("DATASET_ROOT"),
                    help="数据根（默认取 DATASET_ROOT）")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help=f"要训练的模态（默认 {','.join(DEFAULT_CLASSES)}；auto=按数据里出现的）")
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--stride", type=int, default=2, help="体素抽稀步长（越大越快）")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--dry-run", action="store_true", help="只报告精度，不写模型")
    a = ap.parse_args()

    if not a.root:
        print("请给 --root 或设 DATASET_ROOT"); return 2
    classes = tuple(c.strip() for c in a.classes.split(",") if c.strip())
    if len(classes) == 1 and classes[0].lower() == "auto":
        classes = DEFAULT_CLASSES

    print(f"数据根：{a.root}")
    print(f"目标模态：{classes}")
    samples = collect(a.root, classes, a.max_per_class)
    dist = Counter(lab for _, lab in samples)
    print(f"收集到 {len(samples)} 条已标注序列：{dict(dist)}")
    if len(samples) < 30 or len(dist) < 2:
        print("⚠️ 样本不足：确认数据根正确、且 labels/3_serieslabel.xlsx 存在"
              "（或本地模拟集的目录名带模态）"); return 1

    print(f"提取特征（stride={a.stride}, jobs={a.jobs}）……")
    feats: list[np.ndarray] = []
    labels: list[str] = []
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        # pool.map 保序，直接与 samples 对齐（不要按路径回查 —— 那是 O(n²)）
        results = list(pool.map(_features, [(p, a.stride) for p, _ in samples], chunksize=8))
    for (_path, lab), res in zip(samples, results):
        if res is None:
            continue
        feats.append(res[0])
        labels.append(lab)
    if len(feats) < 30:
        print("⚠️ 可用样本不足（读取失败过多）"); return 1
    X = np.stack(feats)
    print(f"特征矩阵 {X.shape}")

    cls_sorted = [c for c in classes if c in dist] or sorted(dist)
    # ---- 交叉验证：准确率 + 混淆矩阵 ----
    rng = np.random.default_rng(0)
    fold_of = np.zeros(len(labels), int)
    for c in cls_sorted:                                          # 分层切折
        idx = np.array([i for i, lab in enumerate(labels) if lab == c])
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % a.folds
    accs: list[float] = []
    cm_total = np.zeros((len(cls_sorted), len(cls_sorted)), int)
    for f in range(a.folds):
        te, tr = fold_of == f, fold_of != f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        model = ModalityModel(classes=cls_sorted).fit(X[tr], np.asarray(labels)[tr])
        pred = model.predict(X[te])
        truth = list(np.asarray(labels)[te])
        accs.append(float(np.mean([p == t for p, t in zip(pred, truth)])))
        cm_total += _confusion(truth, pred, cls_sorted)
    print(f"\n=== {a.folds} 折交叉验证 ===")
    print(f"  准确率：{np.mean(accs):.4f}（各折 {[round(x, 4) for x in accs]}）")
    print("  混淆矩阵（行=真值，列=预测）")
    print("      " + "".join(f"{c:>9}" for c in cls_sorted))
    for i, c in enumerate(cls_sorted):
        row = cm_total[i]
        recall = row[i] / max(1, row.sum())
        print(f"    {c:<8}" + "".join(f"{v:>9}" for v in row) + f"   recall={recall:.3f}")

    model = ModalityModel(classes=cls_sorted).fit(X, np.asarray(labels))
    print("\n=== 特征权重（绝对值最大的前 6 个，用于判断是否学到物理依据）===")
    for ci, c in enumerate(cls_sorted):
        w = model.weights[ci]
        order = np.argsort(-np.abs(w))[:6]
        print(f"  {c:<6}" + ", ".join(f"{FEATURE_NAMES[i]}={w[i]:+.2f}" for i in order))

    if a.dry_run:
        print("\n(--dry-run：未写模型)")
        return 0
    out = model.save(a.out)
    print(f"\n模型已写入 {out}（{out.stat().st_size / 1024:.1f} KB）")
    print("评测期兜底：探针/数据集在序列认不出模态时会尝试用它判别。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
