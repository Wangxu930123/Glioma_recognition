"""目标二-B 的损失与指标（独立副本，可自由修改）。

用**对比式损失**而不是二分类交叉熵，原因是评价指标本身：
规范用 AUC-PR / Recall@10%FPR / Precision@15%Recall —— 都是**排序**指标。
负对数量远多于正对，若用 BCE，模型会把所有对都压到低分（"全都判不重复"），
损失很低但排序能力很差；对比式损失直接优化"正对相似度高于负对"，
与指标同向。
"""
from __future__ import annotations

import numpy as np


def build_loss_fn(cfg: dict):
    """返回 ``loss_fn(out, batch)``；需要 engine 提供 ``embed_a``/``embed_b``。"""
    import torch
    import torch.nn.functional as F

    w = cfg.get("loss_weights") or {}
    margin = float(w.get("margin", 0.3))

    def loss_fn(out, batch):
        ea, eb = out.get("embed_a"), out.get("embed_b")
        if ea is None or eb is None:
            # 配对前向未触发（例如该 batch 没有 image_b）：返回零损失并明确标注，
            # 而不是悄悄用主 embed 顶替——那会让训练目标变成恒等映射。
            z = torch.zeros((), device=next(iter(batch.values())).device, requires_grad=True)
            return z, {"embed": 0.0, "skipped": 1.0}
        sim = F.cosine_similarity(ea.float(), eb.float(), dim=1)
        y = batch["pair"].float().reshape(-1)
        pos = y * (1.0 - sim)
        neg = (1.0 - y) * F.relu(sim - margin)
        loss = (pos + neg).mean()
        return loss, {"embed": float(loss.detach()), "pos_sim": float((sim * y).sum() / y.sum().clamp_min(1e-6))}

    return loss_fn


def evaluate_metrics(model, val_ds, train_ds=None) -> dict:
    """验证集上的排序质量：AUC（正对相似度应高于负对）。"""
    import torch
    import torch.nn.functional as F

    dev = next(model.parameters()).device
    model.eval()
    sims, ys = [], []
    with torch.inference_mode():
        for i in range(len(val_ds)):
            item = val_ds[i]
            a = torch.as_tensor(item["image"])[None].to(dev, torch.float32)
            b = torch.as_tensor(item["image_b"])[None].to(dev, torch.float32)
            ea = F.normalize(model(a)["embed"].float(), dim=1)
            eb = F.normalize(model(b)["embed"].float(), dim=1)
            sims.append(float(F.cosine_similarity(ea, eb, dim=1).item()))
            ys.append(float(np.asarray(item["pair"]).ravel()[0]))
    model.train()
    s, y = np.asarray(sims), np.asarray(ys)
    auc = _auc(s, y)
    return {"auc": auc, "n_pos": int((y > 0.5).sum()), "n": int(len(y)),
            "pos_sim": float(s[y > 0.5].mean()) if (y > 0.5).any() else None,
            "neg_sim": float(s[y <= 0.5].mean()) if (y <= 0.5).any() else None,
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
