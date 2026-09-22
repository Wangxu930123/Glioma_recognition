"""离线评估：折内验证集 Dice/NSD/HD95 + 重复影像三项指标 + 阈值扫描。

用法：
    python -m src.evaluation.evaluate --fold 0 [--ckpt ...] [--limit 20] [--tta]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..data.dataset import load_case_cached, make_targets, resolve_cache_dir
from ..inference.pipeline import GliomaPipeline
from ..inference.duplicate import calibrate, gold_similarities, match_pairs
from ..utils.config import assert_data_source, load_config, load_paths, resolve
from . import metrics as M


def _thr_scan(probs: list[np.ndarray], gts: list[np.ndarray], c: int,
              grid=None) -> tuple[float, float]:
    grid = grid or [round(0.05 * i, 2) for i in range(2, 19)]
    best = (0.5, -1.0)
    for t in grid:
        num = den = 0.0
        for p, g in zip(probs, gts):
            pr, gt = p[c] > t, g[c] > 0.5
            num += 2.0 * float((pr & gt).sum())
            den += float(pr.sum() + gt.sum())
        d = num / den if den > 0 else 1.0
        if d > best[1]:
            best = (float(t), d)
    return best


def eval_fold(fold: int, ckpt: str | list[str], limit: int | None = None) -> dict:
    """评估某一折的 val 集；``ckpt`` 传**列表**即为多折集成（含 TTA 后的均值融合）。"""
    paths = load_paths()
    with open(resolve(paths["folds"]), encoding="utf-8") as f:
        folds = json.load(f)
    with open(resolve(paths["manifest"]), encoding="utf-8") as f:
        man = json.load(f)
    assert_data_source(man, phase="train")                        # 数据源合规闸门
    by_acc = {c["accession"]: c for c in man["cases"]}
    val_cases = [by_acc[a] for a in folds[str(fold)]["val"] if a in by_acc]
    if limit:
        val_cases = val_cases[:limit]

    pre = load_config("preprocess.yaml")
    ckpts = [ckpt] if isinstance(ckpt, str) else [str(c) for c in ckpt]
    pipe = GliomaPipeline(ckpts)
    probs, gts, rows, embeds, feats = [], [], [], {}, {}
    from ..inference.pipeline import postprocess
    from ..inference.writer import case_fingerprint

    cache_dir = resolve_cache_dir()
    for case in val_cases:
        acc = case["accession"]
        try:
            vol, aff, masks = load_case_cached(case, pre, cache_dir)
            gt = make_targets(masks, vol.shape[1:])
            res = pipe.predict_prob(case, vol)
            spacing = tuple(float(np.linalg.norm(aff[:3, i])) for i in range(3))
            pc, pp = postprocess(res["seg"][0] > pipe.thresholds[0],
                                 res["seg"][1] > pipe.thresholds[1],
                                 int(pre["inference"]["min_tumor_voxels"]), spacing)
            probs.append(res["seg"]); gts.append(gt)
            rows.append({"accession": acc,
                         "dice_core": M.dice(pc, gt[0] > 0.5),
                         "dice_peri": M.dice(pp, gt[1] > 0.5),
                         "nsd_core": M.nsd(pc, gt[0] > 0.5, spacing),
                         "nsd_peri": M.nsd(pp, gt[1] > 0.5, spacing),
                         "hd95_peri": M.hd95(pp, gt[1] > 0.5, spacing)})
            embeds[acc] = np.asarray(res["embed"], np.float32)
            feats[acc] = case_fingerprint(case)
        except Exception as e:                                    # noqa: BLE001
            print(f"[eval] ✗ {acc}: {e}")

    rep: dict = {"fold": fold, "n": len(rows), "thresholds": pipe.thresholds}
    if rows:
        for k in ("dice_core", "dice_peri", "nsd_core", "nsd_peri"):
            rep[k] = float(np.mean([r[k] for r in rows]))
        rep["dice_mean"] = 0.5 * (rep["dice_core"] + rep["dice_peri"])
        rep["hd95_peri"] = float(np.nanmean([r["hd95_peri"] for r in rows]))
        if probs:
            rep["thr_scan_core"] = _thr_scan(probs, gts, 0)
            rep["thr_scan_peri"] = _thr_scan(probs, gts, 1)

    gold = {tuple(sorted(p)) for p in (man.get("special", {}).get("gold_pairs") or [])}
    if gold and embeds:
        pos, neg = gold_similarities(embeds, [list(g) for g in gold])
        calib = calibrate(pos, neg)
        w_fp = float(pre["inference"].get("fp_weight", 0.5))
        for w in (0.0, w_fp):
            pairs = match_pairs(embeds, calib, topk=int(pre["inference"]["duplicate_topk"]),
                                feats=feats, w_fp=w)
            rep[f"duplicate_w{w}"] = M.duplicate_report(pairs, gold, sorted(embeds))
        rep["calib"] = calib
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default=None,
                    help="权重路径；**逗号分隔即为多折集成**，如 g4_fold0/best.pth,g4_fold1/best.pth")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    raw = a.ckpt or os.path.join(load_paths()["checkpoints_dir"], f"g4_fold{a.fold}", "best.pth")
    ckpts: list[str] = []
    for c in str(raw).split(","):
        c = c.strip()
        if c:
            ckpts.append(c if os.path.isabs(c) else resolve(c))
    for c in ckpts:
        if not os.path.exists(c):
            raise SystemExit(f"[eval] 权重不存在: {c}")
    print(f"[eval] fold={a.fold}  权重 {len(ckpts)} 个"
          f"{'（多折集成）' if len(ckpts) > 1 else ''}", flush=True)
    for c in ckpts:
        print(f"        · {c}", flush=True)
    eval_fold(a.fold, ckpts, a.limit)


if __name__ == "__main__":
    main()
