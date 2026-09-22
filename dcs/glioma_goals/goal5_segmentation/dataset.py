"""Goal5 Segmentation 的数据集（独立副本，可自由修改）。

继承 ``shared.data.BaseCaseDataset``（负责读盘 / 1mm 公共网格 / patch / 增强），
只实现 :meth:`build_label` —— 本任务的监督信号。

**验证集划分由本 Goal 自己做**（``train.val_ratio``）：这样五个人各自训练时
互不影响，也不需要等别人先产出统一的折划分文件。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from shared.data import (BaseCaseDataset, discover_cases,
                         split_train_val)

GOAL = 'goal5_segmentation'


def split_cases(cases: list[dict], val_ratio: float, seed: int):
    """按 accession 稳定划分 train/val（同一 seed 结果可复现）。"""
    idx = list(range(len(cases)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(len(cases) * float(val_ratio))))
    val_idx = set(idx[:n_val])
    return ([c for i, c in enumerate(cases) if i not in val_idx],
            [c for i, c in enumerate(cases) if i in val_idx])


def build_datasets(cfg: dict, data_root: Path, goal_dir: Path, limit: int = 0,
                   seed: int = 42):
    """构建 ``(train_ds, val_ds)``；缓存写在**本 Goal 目录**下，并行安全。"""
    tr = cfg.get("train") or {}
    dc = cfg.get("data") or {}
    cases = discover_cases(data_root, limit=limit or None)
    # 验证集优先用**统一折划分**（folds.json）。
    # 这条与"五个 Goal 是由五个人并行做、还是一个人串行做"**无关**：
    #   · 一个人串行做时，统一口径才让五个指标互相可比；
    #   · 五个人并行做时，统一口径才让各 Goal 的权重可以合并/集成。
    # 仅当折划分不存在、或与当前数据对不上时才回退到 val_ratio（并有日志提示）。
    train_cases, val_cases, _split_src = split_train_val(
        cases, cfg, data_root, goal_dir, seed)

    cache = (goal_dir / str(dc.get("cache", "cache"))).resolve()
    common = tuple(dc.get("common_spacing", [1.0, 1.0, 1.0]))
    patch = tuple(tr.get("patch", [96, 96, 96]))
    return (
        SegmentationDataset(train_cases, patch=patch, train=True, common_spacing=common,
                 cache_dir=cache, aug_cfg=cfg, seed=seed),
        SegmentationDataset(val_cases, patch=patch, train=False, common_spacing=common,
                 cache_dir=cache, aug_cfg=cfg, seed=seed + 1),
    )


class SegmentationDataset(BaseCaseDataset):
    """Goal5 Segmentation 的数据集。"""

    def build_label(self, case: dict, masks: dict, vol_shape) -> dict:
        """两通道分割标签：通道 0 = core，通道 1 = peri。

        **任务语义上 core ⊆ peri**（增强核心区是周围总异常区的一部分）。
        若数据里只给了 core，则 peri 用 core 兜底——否则 peri 通道会全是背景，
        模型学不到东西。
        """
        d, h, w = vol_shape
        tgt = np.zeros((2, d, h, w), np.float32)
        core = masks.get("core")
        peri = masks.get("peri")
        if core is not None:
            tgt[0] = core.astype(np.float32)
        if peri is not None:
            tgt[1] = peri.astype(np.float32)
        elif core is not None:
            tgt[1] = core.astype(np.float32)
        return {"target": tgt}
