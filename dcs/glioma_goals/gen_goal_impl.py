#!/usr/bin/env python
"""生成六个 Goal 的差异化实现：``model.py`` / ``dataset.py`` / ``losses.py``。

为什么这三件要"每个 Goal 一份副本"（而不是继续共享）：

- ``dataset.py`` 与 ``losses.py`` 承载的是**各任务真正的算法逻辑**——
  目标一要"检查级二分类"，目标四是"14 个字段的多任务分类"，
  目标五是"两通道分割"，它们的标签形状、损失形式、指标定义完全不同；
- 规范也要求每个 Goal 能独立演进。把这三件做成独立副本后，
  队员可以在自己目录里放手修改，**改了不影响别人**。

``model.py`` 虽然六个 Goal 都用同一骨干，但同样给独立副本：
它声明的是"本任务从骨干取哪一路输出"，属于任务语义，不是公共设施。
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

MODEL_TMPL = '''"""{title} 的模型定义（独立副本，可自由修改）。

默认复用 ``shared`` 中的骨干；本文件的价值在于声明**本任务用哪一路输出**，
以及允许你换成完全不同的网络而不影响其他 Goal。
"""
from __future__ import annotations

from shared.backbone_mednext import MedNeXtNet
from shared.backbone_unet3d import GliomaNet

#: 本任务读取的骨干输出（供 losses.py 与 evaluate.py 参照）
HEAD = {head!r}

#: 本任务的分类头定义（仅 {goal} 使用；多任务骨干会同时构建全部头，
#: 但训练时只对本任务那一路回传损失）
CLS_SPEC: list[tuple[str, int]] = {cls_spec!r}


def build_model(cfg: dict):
    """按 config.yaml 构建骨干（结构与 ``shared.factory`` 保持一致，便于互相加载）。"""
    mc = cfg.get("model") or {{}}
    in_ch = int(mc.get("in_channels", 4))
    spec = CLS_SPEC
    if str(mc.get("arch", "mednext")) == "resunet":
        return GliomaNet(in_ch=in_ch, base=int(mc.get("base", 32)), cls_spec=spec)
    return MedNeXtNet(
        in_ch=in_ch, base=int(mc.get("base", 32)), depth=int(mc.get("depth", 4)),
        cls_spec=spec, blocks_per_stage=int(mc.get("blocks_per_stage", 2)),
        k=int(mc.get("k", 3)), expand=int(mc.get("expand", 2)),
        aniso_z=bool(mc.get("aniso_z", False)), max_ch=int(mc.get("max_ch", 320)),
    )
'''

BASE_DATASET = '''"""{title} 的数据集（独立副本，可自由修改）。

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

from shared.data import BaseCaseDataset, discover_cases

GOAL = {goal!r}


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
    tr = cfg.get("train") or {{}}
    dc = cfg.get("data") or {{}}
    cases = discover_cases(data_root, limit=limit or None)
    train_cases, val_cases = split_cases(cases, tr.get("val_ratio", 0.2), seed)

    cache = (goal_dir / str(dc.get("cache", "cache"))).resolve()
    common = tuple(dc.get("common_spacing", [1.0, 1.0, 1.0]))
    patch = tuple(tr.get("patch", [96, 96, 96]))
    return (
        {DS_CLS}(train_cases, patch=patch, train=True, common_spacing=common,
                 cache_dir=cache, aug_cfg=cfg, seed=seed),
        {DS_CLS}(val_cases, patch=patch, train=False, common_spacing=common,
                 cache_dir=cache, aug_cfg=cfg, seed=seed + 1),
    )


class {DS_CLS}(BaseCaseDataset):
    """{title} 的数据集。"""

    def build_label(self, case: dict, masks: dict, vol_shape) -> dict:
        raise NotImplementedError("{title} 的 build_label 未实现")
'''

LOSS_TMPL = '''"""{title} 的损失与指标（独立副本，可自由修改）。"""
from __future__ import annotations

import numpy as np


def build_loss_fn(cfg: dict):
    """返回 ``loss_fn(out, batch) -> (loss_tensor, parts_dict)``。

    ``shared.engine.train`` 会调用它；返回分项便于日志观察是哪一路在退化。
    """
    raise NotImplementedError


def evaluate_metrics(model, val_ds, train_ds=None) -> dict:
    """验证集指标（口径由本 Goal 自己定义）。"""
    raise NotImplementedError
'''

# --------------------------------------------------------------------------- #
# 各 Goal 的差异化实现
# --------------------------------------------------------------------------- #
IMPLS: dict[str, dict] = {
    "goal1_authenticity": {
        "head": "special[0]",
        "cls_spec": [("TumorProbability", 1)],
        "ds_cls": "AuthenticityDataset",
        "dataset": '''
    def build_label(self, case: dict, masks: dict, vol_shape) -> dict:
        """检查级二分类标签：假人体/非人体 = 1。

        标签来源（按优先级）：
        1. 病历目录下的 ``label.json``（``{"fake": 1}`` / ``{"not_human": 1}``）；
        2. 目录名包含 ``fake`` / ``fakehuman`` / ``not_human``。

        **为什么不用"有无掩码"当标签**：真实病例也可能没有掩码（未标注），
        用掩码判断会把"未标注的正常病例"误当作正样本。
        """
        import json as _json
        from pathlib import Path as _P

        y = 0.0
        lj = _P(case["dir"]) / "label.json"
        if lj.is_file():
            d = _json.loads(lj.read_text(encoding="utf-8"))
            y = float(int(d.get("fake", d.get("not_human", 0))))
        else:
            name = case["accession"].lower()
            y = 1.0 if any(k in name for k in ("fake", "nothuman", "not_human")) else 0.0
        return {"special_target": np.array([y], np.float32),
                "special_mask": np.ones(1, np.float32)}
''',
        "losses": '''
def build_loss_fn(cfg: dict):
    """检查级二分类损失（BCE）。"""
    import torch
    import torch.nn.functional as F

    def loss_fn(out, batch):
        logits = out["special"].float()[:, :1]                    # 只取本任务那一路
        target = batch["special_target"].float()[:, :1]
        mask = batch.get("special_mask")
        m = mask.float()[:, :1] if mask is not None else torch.ones_like(target)
        per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (per * m).sum() / m.sum().clamp_min(1e-6)
        return loss, {"special": float(loss.detach())}

    return loss_fn


def evaluate_metrics(model, val_ds, train_ds=None) -> dict:
    """ROC-AUC（目标一用 AUC 评估）。"""
    import torch

    dev = next(model.parameters()).device
    model.eval()
    scores, labels = [], []
    with torch.inference_mode():
        for i in range(len(val_ds)):
            item = val_ds[i]
            x = torch.as_tensor(item["image"])[None].to(dev, torch.float32)
            out = model(x)
            scores.append(float(torch.sigmoid(out["special"].float()[:, 0]).item()))
            labels.append(float(np.asarray(item["special_target"]).ravel()[0]))
    model.train()
    s, y = np.asarray(scores), np.asarray(labels)
    auc = _auc(s, y)
    return {"auc": auc, "n_pos": int((y > 0.5).sum()), "n": int(len(y)), "score": auc}


def _auc(scores, labels) -> float:
    """ROC-AUC（Mann-Whitney U，含并列处理）。"""
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r = ranks[: len(pos)].sum()
    return float((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
''',
    },
    "goal2_stitched": {
        "head": "special[1]",
        "cls_spec": [("TumorProbability", 1)],
        "ds_cls": "StitchedDataset",
        "dataset": '''
    def build_label(self, case: dict, masks: dict, vol_shape) -> dict:
        """拼接影像标签（``label.json`` 的 ``stitched`` 字段，或目录名含 composition/stitch）。"""
        import json as _json
        from pathlib import Path as _P

        y = 0.0
        lj = _P(case["dir"]) / "label.json"
        if lj.is_file():
            d = _json.loads(lj.read_text(encoding="utf-8"))
            y = float(int(d.get("stitched", d.get("composition", 0))))
        else:
            name = case["accession"].lower()
            y = 1.0 if any(k in name for k in ("comp", "stitch", "拼接")) else 0.0
        return {"special_target": np.array([y], np.float32),
                "special_mask": np.ones(1, np.float32)}
''',
        "losses": '''
def build_loss_fn(cfg: dict):
    """拼接二分类损失（取 special 的**第 2 通道**，与目标一互不干扰）。"""
    import torch
    import torch.nn.functional as F

    def loss_fn(out, batch):
        logits = out["special"].float()[:, 1:2]
        target = batch["special_target"].float()[:, :1]
        mask = batch.get("special_mask")
        m = mask.float()[:, :1] if mask is not None else torch.ones_like(target)
        per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (per * m).sum() / m.sum().clamp_min(1e-6)
        return loss, {"special": float(loss.detach())}

    return loss_fn


def evaluate_metrics(model, val_ds, train_ds=None) -> dict:
    """ROC-AUC。"""
    import torch

    dev = next(model.parameters()).device
    model.eval()
    scores, labels = [], []
    with torch.inference_mode():
        for i in range(len(val_ds)):
            item = val_ds[i]
            x = torch.as_tensor(item["image"])[None].to(dev, torch.float32)
            out = model(x)
            scores.append(float(torch.sigmoid(out["special"].float()[:, 1]).item()))
            labels.append(float(np.asarray(item["special_target"]).ravel()[0]))
    model.train()
    s, y = np.asarray(scores), np.asarray(labels)
    auc = _auc(s, y)
    return {"auc": auc, "n_pos": int((y > 0.5).sum()), "n": int(len(y)), "score": auc}


def _auc(scores, labels) -> float:
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r = ranks[: len(pos)].sum()
    return float((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
''',
    },
    "goal3_tumor": {
        "head": "cls[TumorProbability]",
        "cls_spec": [("TumorProbability", 1)],
        "ds_cls": "TumorDataset",
        "dataset": '''
    def build_label(self, case: dict, masks: dict, vol_shape) -> dict:
        """胶质瘤二分类标签。

        ⚠️ **本任务最大的障碍是数据**：本地/公开数据全是胶质瘤，缺少
        "非肿瘤性病变"（脑梗死、脑脓肿等）负样本。当前用"是否有掩码"近似
        （有掩码 → 有肿瘤），这只在负样本补齐前作为占位，
        真实 AUC 无法评估。请务必在 README 的"已知限制"中如实记录。
        """
        y = 1.0 if masks else 0.0
        return {"labels": np.array([y], np.float32),
                "label_mask": np.ones(1, np.float32)}
''',
        "losses": '''
def build_loss_fn(cfg: dict):
    """肿瘤二分类损失（BCE）。"""
    import torch
    import torch.nn.functional as F

    def loss_fn(out, batch):
        rows = out.get("cls") or []
        spec = getattr(out, "cls_spec", None) or []
        logits = None
        for i, (key, _n) in enumerate(spec or [("TumorProbability", 1)]):
            if key == "TumorProbability" and i < len(rows):
                logits = rows[i].float()
                break
        if logits is None:
            raise RuntimeError("骨干输出里没有 TumorProbability 头")
        target = batch["labels"].float()[:, :1]
        m = batch.get("label_mask")
        mm = m.float()[:, :1] if m is not None else torch.ones_like(target)
        per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (per * mm).sum() / mm.sum().clamp_min(1e-6)
        return loss, {"cls": float(loss.detach())}

    return loss_fn


def evaluate_metrics(model, val_ds, train_ds=None) -> dict:
    """ROC-AUC —— **注意**：负样本缺失时该值无统计意义。"""
    import torch

    dev = next(model.parameters()).device
    model.eval()
    scores, labels = [], []
    spec = None
    with torch.inference_mode():
        for i in range(len(val_ds)):
            item = val_ds[i]
            x = torch.as_tensor(item["image"])[None].to(dev, torch.float32)
            out = model(x)
            rows = out.get("cls") or []
            idx = 0
            for j, (k, _n) in enumerate(getattr(out, "cls_spec", None) or [("TumorProbability", 1)]):
                if k == "TumorProbability":
                    idx = j
                    break
            scores.append(float(torch.sigmoid(rows[idx].float().ravel()[0]).item()))
            labels.append(float(np.asarray(item["labels"]).ravel()[0]))
    model.train()
    s, y = np.asarray(scores), np.asarray(labels)
    n_pos, n_neg = int((y > 0.5).sum()), int((y <= 0.5).sum())
    auc = _auc(s, y)
    warn = "" if (n_pos and n_neg) else "（缺负样本，AUC 无统计意义）"
    return {"auc": auc, "n_pos": n_pos, "n_neg": n_neg, "note": warn,
            "score": auc if np.isfinite(auc) else 0.0}


def _auc(scores, labels) -> float:
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r = ranks[: len(pos)].sum()
    return float((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
''',
    },
    "goal4_diagnosis": {
        "head": "cls[14 个字段]",
        "cls_spec": [("TumorProbability", 1), ("Location", 15), ("Morphology", 3),
                     ("WHO_Grade", 4), ("Enhancement", 1), ("EnhancementPattern", 8),
                     ("Necrosis", 1), ("CysticChange", 1), ("Hemorrhage", 1),
                     ("Calcification", 1), ("Margin", 1), ("Lobulation", 1),
                     ("Signal_T2WI", 3), ("Signal_FLAIR", 3)],
        "ds_cls": "DiagnosisDataset",
        "dataset": '''
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
''',
        "losses": '''
def build_loss_fn(cfg: dict):
    """多字段分类损失（二分类 BCE + 多分类 CE，按 label_mask 剔除未标注字段）。"""
    import torch
    import torch.nn.functional as F

    from model import CLS_SPEC

    def loss_fn(out, batch):
        rows = out.get("cls") or []
        labels = batch["labels"].float()
        mask = batch.get("label_mask")
        mm = mask.float() if mask is not None else torch.ones_like(labels)
        total = None
        n_used = 0
        for i, (_key, n_cls) in enumerate(CLS_SPEC):
            if i >= len(rows):
                continue
            m = mm[:, i]
            if float(m.sum()) <= 0:
                continue                                          # 本批该字段全缺 → 跳过
            lg = rows[i].float()
            tgt = labels[:, i]
            if n_cls <= 1:
                per = F.binary_cross_entropy_with_logits(
                    lg.reshape(-1), tgt.clamp(0, 1), reduction="none")
            else:
                per = F.cross_entropy(lg, tgt.long().clamp(0, n_cls - 1), reduction="none")
            term = (per * m).sum() / m.sum().clamp_min(1e-6)
            total = term if total is None else total + term
            n_used += 1
        if total is None:
            total = torch.zeros((), device=labels.device, requires_grad=True)
        return total / max(1, n_used), {"cls": float(total.detach()) / max(1, n_used)}

    return loss_fn


def evaluate_metrics(model, val_ds, train_ds=None) -> dict:
    """各字段准确率（仅统计有标注的样本）。"""
    import torch

    from model import CLS_SPEC

    dev = next(model.parameters()).device
    model.eval()
    hits = {k: [0, 0] for k, _ in CLS_SPEC}
    with torch.inference_mode():
        for i in range(len(val_ds)):
            item = val_ds[i]
            x = torch.as_tensor(item["image"])[None].to(dev, torch.float32)
            out = model(x)
            rows = out.get("cls") or []
            y = np.asarray(item["labels"]).ravel()
            m = np.asarray(item.get("label_mask", np.ones_like(y))).ravel()
            for j, (k, n_cls) in enumerate(CLS_SPEC):
                if j >= len(rows) or m[j] <= 0:
                    continue
                lg = rows[j].float().cpu().numpy().ravel()
                pred = (lg[0] > 0) if n_cls <= 1 else int(np.argmax(lg))
                truth = int(y[j]) if n_cls <= 1 else int(y[j])
                hits[k][1] += 1
                hits[k][0] += int(bool(pred) == bool(truth)) if n_cls <= 1 else int(pred == truth)
    model.train()
    acc = {k: (v[0] / v[1]) for k, v in hits.items() if v[1] > 0}
    mean = float(np.mean(list(acc.values()))) if acc else 0.0
    return {**acc, "mean_acc": mean, "score": mean}
''',
    },
    "goal5_segmentation": {
        "head": "seg[0]=core, seg[1]=peri",
        "cls_spec": [("TumorProbability", 1)],
        "ds_cls": "SegmentationDataset",
        "dataset": '''
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
''',
        "losses": '''
def build_loss_fn(cfg: dict):
    """分割损失：Dice + 边界加权 BCE（体素极度不平衡，纯 BCE 会退化为全背景）。"""
    import torch
    import torch.nn.functional as F

    boost = float((cfg.get("loss_weights") or {}).get("boundary_boost", 2.0))

    def _dice(logits, target):
        p = torch.sigmoid(logits)
        dims = tuple(range(2, p.dim()))
        num = 2 * (p * target).sum(dims) + 1e-6
        den = p.sum(dims) + target.sum(dims) + 1e-6
        return (1 - num / den).mean()

    def _bce(logits, target):
        w = torch.ones_like(target)
        if boost > 1.0:
            t = target.float()
            dil = F.max_pool3d(t, 3, 1, 1)
            ero = -F.max_pool3d(-t, 3, 1, 1)
            w = w + (boost - 1.0) * (dil - ero).clamp(0, 1)
        return F.binary_cross_entropy_with_logits(logits, target, weight=w)

    def loss_fn(out, batch):
        logits = out["seg"].float()
        tgt = batch["target"].float()[:, :logits.shape[1]]
        seg = _dice(logits, tgt) + _bce(logits, tgt)
        total = seg
        parts = {"seg": float(seg.detach())}
        ds = out.get("ds") or []
        if ds:
            acc = None
            for d in ds:
                t = F.interpolate(tgt, size=tuple(d.shape[2:]), mode="nearest")
                term = _dice(d.float(), t) + _bce(d.float(), t)
                acc = term if acc is None else acc + term
            dsl = acc / len(ds)
            w = float((cfg.get("loss_weights") or {}).get("ds", 0.4))
            total = total + w * dsl
            parts["ds"] = float(dsl.detach())
        return total, parts

    return loss_fn


def evaluate_metrics(model, val_ds, train_ds=None) -> dict:
    """验证集 Dice（core / peri 分别统计）。"""
    import torch

    dev = next(model.parameters()).device
    model.eval()
    acc = {"core": [], "peri": []}
    with torch.inference_mode():
        for i in range(len(val_ds)):
            item = val_ds[i]
            x = torch.as_tensor(item["image"])[None].to(dev, torch.float32)
            out = model(x)
            p = torch.sigmoid(out["seg"].float()).cpu().numpy()[0]
            g = np.asarray(item["target"])[:p.shape[0]]
            for c, key in enumerate(("core", "peri")):
                if c >= p.shape[0] or c >= g.shape[0]:
                    continue
                pb, gb = p[c] > 0.5, g[c] > 0.5
                inter = float((pb & gb).sum())
                den = float(pb.sum() + gb.sum())
                acc[key].append(2 * inter / den if den > 0 else float("nan"))
    model.train()
    out = {f"dice_{k}": float(np.nanmean(v)) for k, v in acc.items() if v}
    out["score"] = float(np.mean(list(out.values()))) if out else 0.0
    return out
''',
    },
}


def main() -> int:
    for goal, imp in IMPLS.items():
        d = ROOT / goal
        head, cls_spec, ds_cls = imp["head"], imp["cls_spec"], imp["ds_cls"]
        title = goal.replace("_", " ").title()

        (d / "model.py").write_text(
            MODEL_TMPL.format(title=title, goal=goal, head=head, cls_spec=cls_spec),
            encoding="utf-8")

        base = BASE_DATASET.format(title=title, goal=goal, DS_CLS=ds_cls)
        # 把 NotImplementedError 版本换成该 Goal 的真实实现
        impl = imp["dataset"]
        base = base.replace('''    def build_label(self, case: dict, masks: dict, vol_shape) -> dict:
        raise NotImplementedError("%s 的 build_label 未实现")''' % title,
                              "    #: 字段顺序与 model.CLS_SPEC 严格一致" if False else impl.strip("\n"))
        # 注入 FIELDS/N_CLASSES（Goal4 需要，作为类属性）
        if "FIELDS = [" in impl:
            base = base.replace("    def build_label", impl.strip("\n") + "\n\n    def build_label", 1)
            base = base.replace("\n\n    def build_label\n", "\n", 1)
        (d / "dataset.py").write_text(base, encoding="utf-8")

        (d / "losses.py").write_text(
            LOSS_TMPL.format(title=title).replace(
                '''def build_loss_fn(cfg: dict):
    """返回 ``loss_fn(out, batch) -> (loss_tensor, parts_dict)``。

    ``shared.engine.train`` 会调用它；返回分项便于日志观察是哪一路在退化。
    """
    raise NotImplementedError


def evaluate_metrics(model, val_ds, train_ds=None) -> dict:
    """验证集指标（口径由本 Goal 自己定义）。"""
    raise NotImplementedError''',
                imp["losses"].strip("\n")),
            encoding="utf-8")
        print(f"  ✓ {goal:22s} model.py / dataset.py / losses.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
