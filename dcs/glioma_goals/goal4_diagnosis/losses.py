"""Goal4 Diagnosis 的损失与指标（独立副本，可自由修改）。"""
from __future__ import annotations

import numpy as np


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
            # 本批所有字段都没有标注 → 无监督信号。
            # 这里必须**显式告警**：静默返回 0 损失会让训练日志看起来正常
            # （loss 一路下降），实际上是模型根本没在学，最后拿到一个没用的权重。
            if not getattr(loss_fn, "_warned", False):
                print("[goal4] ⚠️ 本批 14 个字段**全部无标注**（缺 label.json？）→ "
                      "该步无监督信号，模型不会更新。请检查数据是否包含结构化金标准。",
                      flush=True)
                loss_fn._warned = True
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
