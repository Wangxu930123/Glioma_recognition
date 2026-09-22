"""Goal5 Segmentation 的损失与指标（独立副本，可自由修改）。"""
from __future__ import annotations

import numpy as np


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
