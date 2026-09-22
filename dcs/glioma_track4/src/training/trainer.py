"""训练器（赛道4）：多任务损失 + AMP/EMA + 断点续训 + 规范 JSONL 日志 + 多折。

与早期版本的关键差异（每一条都直接对应分数）：
1. **目标一/二真正被训练**：``SpecialImageDataset`` 提供 fake / Composition 监督，
   ``special`` 头不再是从未被优化的随机层（目标一二不达标直接淘汰）；
2. **重复影像嵌入真正被训练**：``DuplicatePairDataset`` 提供金标准正对与随机负对，
   早期 ``batch["pair"]``/``out["embed_a"]`` 从来不存在，嵌入损失恒被跳过；
3. **全局头与推理同尺度**：分类 / 特殊 / 嵌入统一使用"固定物理尺寸整脑视图"，
   早期训练用局部 patch、推理用整脑降采样，尺度不一致导致结构头退化；
4. 稀疏结构化标签用 ``label_mask`` 只回传有效样本；类别不平衡用 BCE 正类加权；
5. 边界加权 + core⊆peri 集合约束；EMA；OneCycle；断点续训；
6. 验证集**每通道阈值搜索**并写入 checkpoint（推理直接使用，避免固定 0.5 的损失）。

用法：
    python -m src.training.trainer --config train --fold 0 --tag g4_fold0
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import (DuplicatePairDataset, GliomaDataset, SpecialImageDataset,
                            build_case_volume, build_folds, make_targets)
from ..models.unet3d import build_model, cls_spec_from_config
from ..utils.config import (assert_data_source, data_source_tag, load_config,
                             load_paths, resolve)
from ..utils.logger import run_logger


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3, 4)) + eps
    den = p.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + eps
    return 1.0 - (num / den).mean()


def _dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool3d(m, k, 1, k // 2)


def _erode(m: torch.Tensor, k: int) -> torch.Tensor:
    return -F.max_pool3d(-m, k, 1, k // 2)


def boundary_weight(target: torch.Tensor, k: int = 5) -> torch.Tensor:
    """边界带权重：目标轮廓附近加权（改善 NSD/HD95，对 Dice 也有轻微正收益）。"""
    band = _dilate(target, k) - _erode(target, k)
    return 1.0 + 2.0 * band


def seg_loss(logits: torch.Tensor, target: torch.Tensor, w_ce: float = 1.0,
             bnd: float = 0.0, pos_weight: float = 1.0) -> torch.Tensor:
    """Dice + 加权 BCE（+ 可选边界加权）。深监督输出尺寸不同 → 目标最近邻下采样。"""
    if logits.shape[2:] != target.shape[2:]:
        target = F.interpolate(target, size=logits.shape[2:], mode="nearest")
    l = dice_loss(logits, target)
    if w_ce > 0:
        w = torch.ones_like(target)
        if bnd > 0:
            w = w + bnd * (boundary_weight(target) - 1.0)
        pw = torch.tensor([pos_weight], device=logits.device)
        l = l + w_ce * F.binary_cross_entropy_with_logits(logits, target, weight=w, pos_weight=pw)
    return l


def contain_loss(seg_logits: torch.Tensor) -> torch.Tensor:
    """core 应包含于 peri：惩罚 core 有而 peri 无的体素。"""
    p = torch.sigmoid(seg_logits)
    return (p[:, 0] * (1 - p[:, 1])).mean()


def seg_total_loss(out: dict, target: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    """分割总损失 = Dice + 加权 BCE（+ 深监督 + core⊆peri 集合约束）。"""
    lw = cfg["loss"]
    parts: dict[str, float] = {}
    total = seg_loss(out["seg"], target, w_ce=lw.get("ce", 1.0),
                     bnd=lw.get("boundary", 0.0), pos_weight=lw.get("seg_pos_weight", 1.0))
    if out.get("ds") and lw.get("deep_sup", 0) > 0:
        ds = sum(seg_loss(d, target, w_ce=lw.get("ce", 1.0)) for d in out["ds"]) / len(out["ds"])
        total = total + lw["deep_sup"] * ds
        parts["ds"] = float(ds)
    if lw.get("contain", 0) > 0:
        total = total + lw["contain"] * contain_loss(out["seg"])
    parts["seg"] = float(total)
    return total, parts


def cls_total_loss(out: dict, labels: torch.Tensor, label_mask: torch.Tensor, cfg: dict,
                   cls_spec: list[tuple[str, int]]) -> tuple[torch.Tensor, dict]:
    """结构化字段损失（稀疏 ``label_mask``：只在有金标准的样本上回传）。"""
    lw = cfg["loss"]
    cl = torch.zeros((), device=labels.device)
    n_used = 0
    for i, (_name, n_cls) in enumerate(cls_spec):
        if i >= len(out["cls"]):
            break
        m = label_mask[:, i] > 0
        if int(m.sum()) == 0:
            continue
        logit, y = out["cls"][i][m], labels[m, i]
        if n_cls == 1:
            pw = torch.tensor([float(lw.get("cls_pos_weight", 2.0))], device=logit.device)
            cl = cl + F.binary_cross_entropy_with_logits(logit.squeeze(1), y, pos_weight=pw)
        else:
            cl = cl + F.cross_entropy(logit, y.long(), label_smoothing=0.05)
        n_used += 1
    if n_used:
        cl = cl / n_used
    return lw.get("cls", 1.0) * cl, {"cls": float(cl)}


def special_total_loss(out: dict, target: torch.Tensor, label_mask: torch.Tensor,
                       cfg: dict) -> tuple[torch.Tensor, dict]:
    """目标一/二：假人体 / 拼接（逐头 ``label_mask``，避免互相当作负样本）。"""
    logit = out["special"]
    raw = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    sl = (raw * label_mask).sum() / label_mask.sum().clamp_min(1.0)
    return cfg["loss"].get("special", 1.0) * sl, {"special": float(sl)}


def embed_total_loss(out: dict, pair: tuple, cfg: dict) -> tuple[torch.Tensor, dict]:
    """重复影像嵌入：配对余弦相似度的 BCE（正对小 → 相似度高）。"""
    a, b, y = pair
    emb = F.normalize(out["embed"], dim=1)
    na = a.shape[0]
    sim = (emb[:na] * emb[na:]).sum(dim=1)
    el = F.binary_cross_entropy_with_logits(sim * float(cfg["loss"].get("embed_scale", 10.0)), y)
    return cfg["loss"].get("embed", 0.3) * el, {"embed": float(el)}


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
@torch.no_grad()
def dice_metric(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> tuple[float, float]:
    p = (torch.sigmoid(logits) > thr).float()
    out = []
    for c in range(target.shape[1]):
        num = 2 * (p[:, c] * target[:, c]).sum() + 1e-5
        den = p[:, c].sum() + target[:, c].sum() + 1e-5
        out.append(float(num / den))
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def _load_manifest(path: str) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return json.load(f)


def _split(cases: list[dict], folds: dict, fold: int):
    fd = folds[str(fold)]
    by_acc = {c["accession"]: c for c in cases}
    tr = [by_acc[a] for a in fd["train"] if a in by_acc]
    va = [by_acc[a] for a in fd["val"] if a in by_acc]
    return tr, va


def make_loaders(cfg: dict, fold: int):
    paths = load_paths()
    man = _load_manifest(paths["manifest"])
    assert_data_source(man, phase="train")                        # 数据源合规闸门
    cases = man["cases"]
    folds_path = resolve(paths["folds"])
    if not os.path.exists(folds_path):
        folds = build_folds(paths["manifest"], n_folds=cfg["folds"], val_ratio=cfg["val_ratio"])
    else:
        with open(folds_path, encoding="utf-8") as f:
            folds = json.load(f)
    tr, va = _split(cases, folds, fold)

    labels_cfg = load_config("labels.yaml")
    fields = labels_cfg["fields"]
    pre = load_config("preprocess.yaml")
    patch = tuple(cfg["patch_size"])
    dl_tr = DataLoader(
        GliomaDataset(tr, train=True, patch=patch, pos_ratio=cfg["sampler"]["pos_ratio"],
                      pre_cfg=pre, label_fields=fields, seed=cfg["seed"], aug_cfg=cfg),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
        drop_last=True, persistent_workers=cfg["num_workers"] > 0)
    dl_va = DataLoader(
        GliomaDataset(va, train=False, patch=patch, pre_cfg=pre, label_fields=fields,
                      seed=cfg["seed"], aug_cfg=cfg),
        batch_size=1, shuffle=False, num_workers=max(1, cfg["num_workers"] // 2))

    # ---- 目标一/二：特殊影像（fake / Composition）----
    special = man.get("special") or {}
    pos_fake = set(special.get("fake_cases") or special.get("fake") or [])
    pos_comp = set(special.get("composition_cases") or special.get("composition") or [])
    aux: dict = {"special": None, "pair": None, "pos_counts": [len(pos_fake), len(pos_comp)]}
    n_spec = int(cfg.get("aux", {}).get("special_batch", 2))
    if (pos_fake or pos_comp) and n_spec > 0:
        ds_sp = SpecialImageDataset(cases, pos_fake, pos_comp, pre_cfg=pre, aug_cfg=cfg,
                                    seed=cfg["seed"] + fold, n_per_epoch=10 ** 6)
        aux["special"] = DataLoader(ds_sp, batch_size=n_spec, shuffle=False,
                                    num_workers=0, drop_last=True)

    # ---- 重复影像嵌入 ----
    gold = [list(p) for p in (special.get("gold_pairs") or [])]
    n_pair = int(cfg.get("aux", {}).get("pair_batch", 2))
    if gold and n_pair > 0:
        ds_pr = DuplicatePairDataset(cases, gold, n_neg_per_pos=cfg.get("aux", {}).get("neg_per_pos", 3),
                                     seed=cfg["seed"] + fold, pre_cfg=pre, aug_cfg=cfg)
        aux["pair"] = DataLoader(ds_pr, batch_size=n_pair, shuffle=True, num_workers=0,
                                 drop_last=True)
    return dl_tr, dl_va, len(tr), len(va), aux


# --------------------------------------------------------------------------- #
# 验证（含**每通道阈值搜索**）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, dl, cfg) -> dict:
    model.eval()
    probs, gts = [], []
    for batch in dl:
        x = batch["image"].cuda(non_blocking=True).float()
        with torch.autocast("cuda", dtype=getattr(torch, cfg.get("amp_dtype", "bfloat16")),
                            enabled=torch.cuda.is_available()):
            out = model(x)
        probs.append(torch.sigmoid(out["seg"].float())[0].cpu().numpy())
        gts.append(batch["target"][0].numpy())
    if not probs:
        return {"dice_core": 0.0, "dice_peri": 0.0, "dice_mean": 0.0, "thresholds": [0.5, 0.5]}

    grid = cfg.get("threshold_grid") or [round(0.05 * i, 2) for i in range(1, 20)]
    best_thr, dices = [], []
    for c in range(2):
        bc, bd = 0.5, -1.0
        for t in grid:
            num = den = 0.0
            for p, g in zip(probs, gts):
                pred = p[c] > t
                gt = g[c] > 0.5
                num += 2.0 * float((pred & gt).sum())
                den += float(pred.sum() + gt.sum())
            d = num / den if den > 0 else 1.0
            if d > bd:
                bc, bd = float(t), d
        best_thr.append(bc)
        dices.append(bd)
    n = len(probs)
    if cfg.get("verbose_eval", False):                            # 全网格 Dice（诊断用）
        pass
    return {"dice_core": dices[0], "dice_peri": dices[1],
            "dice_mean": 0.5 * (dices[0] + dices[1]), "thresholds": best_thr, "n_val": n}


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train(cfg_path: str, fold: int, tag: str | None, no_resume: bool = False,
          pretrained: str | None = None) -> str:
    cfg = load_config(os.path.basename(cfg_path) if cfg_path.endswith(".yaml") else cfg_path)
    paths = load_paths()
    tag = tag or f"g4_fold{fold}"
    out_dir = resolve(os.path.join(paths["checkpoints_dir"], tag))
    os.makedirs(out_dir, exist_ok=True)
    logger = run_logger(resolve(paths["logs_dir"]), tag)

    torch.manual_seed(cfg["seed"] + fold)
    np.random.seed(cfg["seed"] + fold)

    labels_cfg = load_config("labels.yaml")
    cls_spec = cls_spec_from_config(labels_cfg)
    mcfg = cfg.get("model") or {}
    model = build_model(cls_spec, in_ch=len(load_config("preprocess.yaml")["channels"]),
                        arch=mcfg.get("arch", "mednext"), base=int(mcfg.get("base", 32)),
                        depth=int(mcfg.get("depth", 4)),
                        blocks_per_stage=int(mcfg.get("blocks_per_stage", 2)),
                        k=int(mcfg.get("k", 3)), expand=int(mcfg.get("expand", 2)),
                        dropout=float(mcfg.get("dropout", 0.0)),
                        aniso_z=bool(mcfg.get("aniso_z", False)),
                        max_ch=int(mcfg.get("max_ch", 320)),
                        plain_stages=int(mcfg.get("plain_stages", 0)),
                        dec_blocks=int(mcfg.get("dec_blocks", 1))).cuda()
    model.train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    dl_tr, dl_va, n_tr, n_va, aux = make_loaders(cfg, fold)
    steps_per_epoch = max(1, len(dl_tr))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch,
        pct_start=0.05, anneal_strategy="cos")

    start_epoch, best, best_thr = 0, -1.0, [0.5, 0.5]
    last_p = os.path.join(out_dir, "last.pth")
    loaded_pretrained = ""
    if no_resume:
        print(f"[trainer] --no-resume：忽略 {last_p}", flush=True)
    elif os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("model_ema"):
            ema.load_state_dict(ck["model_ema"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        best = float(ck.get("best_metric", -1.0))
        best_thr = list(ck.get("thresholds") or best_thr)
        loaded_pretrained = ck.get("pretrained_from", "")
        print(f"[trainer] 断点续训：epoch {start_epoch}/{cfg['epochs']}", flush=True)
    elif pretrained and os.path.exists(resolve(pretrained)):
        ck = torch.load(resolve(pretrained), map_location="cpu", weights_only=False)
        sd = ck.get("model_ema") or ck.get("model") or ck
        missing = model.load_state_dict(sd, strict=False)
        loaded_pretrained = pretrained
        print(f"[trainer] 加载预训练权重 {pretrained}（缺失 {len(missing.missing_keys)} 键）", flush=True)
    else:
        print("[trainer] 从零训练：未加载预训练权重", flush=True)

    # data_source 按实际数据根生成（官方挂载 → official/*；本地/公开数据 → local/*），
    # 避免把本地验证数据记成官方数据（规范要求日志可追溯数据来源）
    ds_tag_tr = data_source_tag(phase="train")
    ds_tag_va = data_source_tag(phase="val")
    print(f"[trainer] 数据源标识: train={ds_tag_tr} val={ds_tag_va}", flush=True)
    amp_dtype = getattr(torch, cfg.get("amp_dtype", "bfloat16"))
    amp_on = torch.cuda.is_available()
    g_every = max(1, int((cfg.get("global_view") or {}).get("every", 4)))
    log_every = max(1, int(cfg.get("log_every", 20)))
    n_spec_b = int((cfg.get("aux") or {}).get("special_batch", 2)) if aux.get("special") else 0
    n_pair_b = int((cfg.get("aux") or {}).get("pair_batch", 2)) if aux.get("pair") else 0
    sp_iter = itertools.cycle(aux["special"]) if aux.get("special") else None
    pr_iter = itertools.cycle(aux["pair"]) if aux.get("pair") else None
    print(f"[trainer] tag={tag} fold={fold} train={n_tr} val={n_va} steps/epoch={steps_per_epoch} "
          f"特殊影像正样本={aux['pos_counts']} 重复对={aux.get('pair') is not None}", flush=True)

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        run = 0.0
        for step, batch in enumerate(dl_tr):
            x = batch["image"].cuda(non_blocking=True).float()
            y = batch["target"].cuda(non_blocking=True).float()
            batch["labels"] = batch["labels"].cuda().float()
            batch["label_mask"] = batch["label_mask"].cuda().float()

            # ---- 分阶段前向 + 梯度累积（显存峰值 = 单阶段峰值，数学等价于大 batch）----
            opt.zero_grad(set_to_none=True)
            parts: dict[str, float] = {}
            total = torch.zeros((), device=x.device)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                out = model(x)
                l_seg, p_seg = seg_total_loss(out, y, cfg)
            l_seg.backward()
            total = total + l_seg.detach()
            parts.update(p_seg)

            do_global = step % g_every == 0
            if do_global:
                # (b) 主视图 → 结构化字段（与推理同尺度的整脑视图）
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                    gout = model(batch["whole"].cuda(non_blocking=True).float())
                    l_cls, p_cls = cls_total_loss(gout, batch["labels"], batch["label_mask"],
                                                  cfg, cls_spec)
                if l_cls.requires_grad:
                    (l_cls / g_every).backward()
                total = total + l_cls.detach() / g_every
                parts.update(p_cls)
                del gout

            if sp_iter is not None and do_global:
                # (c) 特殊影像（目标一/二）——**独立前向**，避免与主损失争显存
                sb = next(sp_iter)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                    gs = model(sb["image"].cuda(non_blocking=True).float())
                    l_sp, p_sp = special_total_loss(gs, sb["target"].cuda().float(),
                                                    sb["label_mask"].cuda().float(), cfg)
                (l_sp / g_every).backward()
                total = total + l_sp.detach() / g_every
                parts.update(p_sp)
                del gs

            if pr_iter is not None and do_global:
                # (d) 重复影像配对嵌入
                pb = next(pr_iter)
                pair_x = torch.cat([pb["a"].cuda(non_blocking=True).float(),
                                    pb["b"].cuda(non_blocking=True).float()], dim=0)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                    gp = model(pair_x)
                    l_emb, p_emb = embed_total_loss(gp, (pb["a"], pb["b"],
                                                         pb["label"].cuda().float()), cfg)
                (l_emb / g_every).backward()
                total = total + l_emb.detach() / g_every
                parts.update(p_emb)
                del gp

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            losses = {"loss": total, "parts": parts}
            if sched.last_epoch < cfg["epochs"] * steps_per_epoch - 1:
                sched.step()
            with torch.no_grad():                       # EMA
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(cfg["ema_decay"]).add_(pm.detach(), alpha=1 - cfg["ema_decay"])
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            run += float(losses["loss"])
            if step % log_every == 0 and step > 0:
                logger.log(epoch=epoch, step=epoch * steps_per_epoch + step + 1, phase="train",
                           loss=float(losses["loss"]), lr=float(opt.param_groups[0]["lr"]),
                           mode="training", data_source=ds_tag_tr,
                           checkpoint=os.path.join(out_dir, "best.pth"),
                           pretrained_from=loaded_pretrained or None,
                           parts={k: round(v, 4) for k, v in losses["parts"].items()})
                print(f"[{tag}] ep{epoch} step{step+1}/{steps_per_epoch} "
                      f"loss={float(losses['loss']):.4f} "
                      f"{ {k: round(v, 3) for k, v in losses['parts'].items()} }", flush=True)

        vm = {"dice_core": 0.0, "dice_peri": 0.0, "dice_mean": 0.0, "thresholds": best_thr}
        if (epoch + 1) % cfg["val_every"] == 0 and n_va > 0:
            vm = validate(ema, dl_va, cfg)
            ema.eval()                                              # validate 会切 train()，恢复
            logger.log(epoch=epoch, step=(epoch + 1) * steps_per_epoch, phase="val",
                       loss=float(run / max(1, steps_per_epoch)), lr=float(opt.param_groups[0]["lr"]),
                       mode="training", data_source=ds_tag_va,
                       checkpoint=os.path.join(out_dir, "best.pth"),
                       dice_core=vm["dice_core"], dice_peri=vm["dice_peri"],
                       thresholds=vm["thresholds"])
        metric = vm["dice_mean"] if str(cfg.get("checkpoint_metric", "")).startswith("val_dice_mean") \
            else vm["dice_peri"]
        print(f"[{tag}] epoch {epoch} loss={run/max(1,steps_per_epoch):.4f} "
              f"dice_core={vm['dice_core']:.4f} dice_peri={vm['dice_peri']:.4f} "
              f"thr={vm.get('thresholds')} ({time.time()-t0:.0f}s)", flush=True)
        if metric > best:
            best = metric
            best_thr = list(vm.get("thresholds") or best_thr)
            torch.save({"model": model.state_dict(), "model_ema": ema.state_dict(),
                        "cls_spec": cls_spec, "epoch": epoch, "best_metric": best,
                        "thresholds": best_thr, "arch": mcfg.get("arch", "mednext"),
                        "model_cfg": mcfg, "global_size": int((cfg.get("global_view") or {}).get("out", 96)),
                        "global_size_mm": float((cfg.get("global_view") or {}).get("size_mm", 192)),
                        "pretrained_from": loaded_pretrained,
                        "special_trained": bool(aux.get("special")),
                        "config": cfg}, os.path.join(out_dir, "best.pth"))
        torch.save({"model": model.state_dict(), "model_ema": ema.state_dict(),
                    "cls_spec": cls_spec, "epoch": epoch, "best_metric": best,
                    "thresholds": best_thr, "arch": mcfg.get("arch", "mednext"),
                    "model_cfg": mcfg, "pretrained_from": loaded_pretrained,
                    "special_trained": bool(aux.get("special")), "optimizer": opt.state_dict(), "config": cfg}, last_p)
    print(f"[trainer] done. best={best:.4f} thr={best_thr} -> {out_dir}/best.pth", flush=True)
    return os.path.join(out_dir, "best.pth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--pretrained", default=None, help="预训练权重（可选；合规性自行确认）")
    a = ap.parse_args()
    train(a.config, a.fold, a.tag, a.no_resume, a.pretrained)


if __name__ == "__main__":
    main()
