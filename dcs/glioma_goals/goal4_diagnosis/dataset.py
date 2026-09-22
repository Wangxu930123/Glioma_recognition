"""Goal4 Diagnosis 的数据集（独立副本，可自由修改）。

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

GOAL = 'goal4_diagnosis'


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
        DiagnosisDataset(train_cases, patch=patch, train=True, common_spacing=common,
                 cache_dir=cache, aug_cfg=cfg, seed=seed),
        DiagnosisDataset(val_cases, patch=patch, train=False, common_spacing=common,
                 cache_dir=cache, aug_cfg=cfg, seed=seed + 1),
    )


class DiagnosisDataset(BaseCaseDataset):
    """Goal4 Diagnosis 的数据集。"""

    #: 字段顺序与 model.CLS_SPEC 严格一致（顺序错位 = 标签与头错配，且不会报错）
    FIELDS = ["TumorProbability", "Location", "Morphology", "WHO_Grade", "Enhancement",
              "EnhancementPattern", "Necrosis", "CysticChange", "Hemorrhage",
              "Calcification", "Margin", "Lobulation", "Signal_T2WI", "Signal_FLAIR"]

    #: 多分类字段的类别数（二分类为 1）
    N_CLASSES = {1: 1, 2: 15, 3: 3, 4: 4, 5: 1, 6: 8, 7: 1, 8: 1, 9: 1, 10: 1,
                 11: 1, 12: 1, 13: 3, 14: 3}

    #: 字段顺序与 model.CLS_SPEC 严格一致（顺序错位 = 标签与头错配，且不会报错）
    FIELDS = ["TumorProbability", "Location", "Morphology", "WHO_Grade", "Enhancement",
              "EnhancementPattern", "Necrosis", "CysticChange", "Hemorrhage",
              "Calcification", "Margin", "Lobulation", "Signal_T2WI", "Signal_FLAIR"]

    #: 多分类字段的类别数（二分类为 1）
    N_CLASSES = {1: 1, 2: 15, 3: 3, 4: 4, 5: 1, 6: 8, 7: 1, 8: 1, 9: 1, 10: 1,
                 11: 1, 12: 1, 13: 3, 14: 3}

    def build_label(self, case: dict, masks: dict, vol_shape) -> dict:
        """从 ``label.json`` 读 14 个结构化字段。

        **缺失字段必须 mask 掉**（``label_mask=0``）：真实金标准里 14 个字段
        往往只标注了一部分，若把"没标"当成"阴性"，模型会被强迫学 0，
        这类污染在训练日志里完全看不出来，只会让指标莫名偏低。
        """
        import json as _json
        from pathlib import Path as _P

        labels = np.zeros(len(self.FIELDS), np.float32)
        mask = np.zeros(len(self.FIELDS), np.float32)
        lj = _P(case["dir"]) / "label.json"
        raw = {}
        if lj.is_file():
            raw = _json.loads(lj.read_text(encoding="utf-8")) or {}

        for i, f in enumerate(self.FIELDS, start=1):
            if f not in raw or raw[f] is None:
                continue
            v = raw[f]
            n = int(self.N_CLASSES.get(i, 1))
            if n <= 1:
                labels[i - 1] = 1.0 if (v in (True, 1, "1", "yes", "有", "present")) else 0.0
            else:
                labels[i - 1] = float(int(v))
            mask[i - 1] = 1.0
        return {"labels": labels, "label_mask": mask}

    def build_label(self, case: dict, masks: dict, vol_shape) -> dict:
        """从 ``label.json`` 读 14 个结构化字段。

        **缺失字段必须 mask 掉**（``label_mask=0``）：真实金标准里 14 个字段
        往往只标注了一部分，若把"没标"当成"阴性"，模型会被强迫学 0，
        这类污染在训练日志里完全看不出来，只会让指标莫名偏低。
        """
        import json as _json
        from pathlib import Path as _P

        labels = np.zeros(len(self.FIELDS), np.float32)
        mask = np.zeros(len(self.FIELDS), np.float32)
        lj = _P(case["dir"]) / "label.json"
        raw = {}
        if lj.is_file():
            raw = _json.loads(lj.read_text(encoding="utf-8")) or {}

        for i, f in enumerate(self.FIELDS, start=1):
            if f not in raw or raw[f] is None:
                continue
            v = raw[f]
            n = int(self.N_CLASSES.get(i, 1))
            if n <= 1:
                labels[i - 1] = 1.0 if (v in (True, 1, "1", "yes", "有", "present")) else 0.0
            else:
                labels[i - 1] = float(int(v))
            mask[i - 1] = 1.0
        return {"labels": labels, "label_mask": mask}
