"""Goal1 Authenticity 的损失与指标（独立副本，可自由修改）。"""
from __future__ import annotations

import numpy as np


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
