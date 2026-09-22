"""Goal3 Tumor 的损失与指标（独立副本，可自由修改）。"""
from __future__ import annotations

import numpy as np


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
