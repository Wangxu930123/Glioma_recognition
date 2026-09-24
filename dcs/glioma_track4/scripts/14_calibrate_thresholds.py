#!/usr/bin/env python
"""全图阈值标定：用训练好的权重在**全图滑窗推理**上搜索每通道最优阈值并写回 checkpoint。

为什么需要它：训练时为了避免每 epoch 做全图推理（125 例 × ~20s），
`trainer.validate` 只在"病灶中心 patch"上搜阈值。实测该阈值与全图最优阈值
可差 0.05~0.10 Dice（patch 内病灶占比高 → 最优阈值偏低）。本脚本用真正的
滑窗推理重搜，直接提升提交时的 Dice/NSD。

标定集（``--split``）：

- ``external``：**官方验证集**（``data/manifest_val.json``，``01_probe.sh --val`` 生成）。
  验证集与训练集无交集 → 用**全折集成**标定（默认权重=全部折），这是最终口径；
- ``fold``：折内 val（旧口径）；
- ``auto``（默认）：验证集清单可用就用 external，否则回退 fold。

用法：
    python scripts/14_calibrate_thresholds.py --split auto [--fold 0] [--ckpt ...] [--write]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=("auto", "fold", "external"), default="auto",
                    help="auto（默认）=有验证集清单就在官方验证集上标定（全折集成），"
                         "否则用折内 val；fold=强制折内 val；external=强制官方验证集")
    ap.add_argument("--ckpt", default=None,
                    help="逗号分隔即为多折集成；集成时须用集成模型重新标定，"
                         "否则 load_ensemble 取各折阈值均值并非最优"
                         "（--split external 且未指定时=自动取全部折权重）")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部验证集")
    ap.add_argument("--grid", default="0.15,0.25,0.35,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    ap.add_argument("--write", action="store_true", help="把新阈值写回 checkpoint")
    a = ap.parse_args()

    from src.data.dataset import load_case_cached, make_targets, resolve_cache_dir
    from src.inference.pipeline import GliomaPipeline
    from src.utils.config import (external_val_manifest, fold_ckpts, load_config,
                                  load_paths, resolve)

    paths = load_paths()
    split = a.split
    if split == "auto":
        split = "external" if external_val_manifest()[0] is not None else "fold"
    raw = a.ckpt
    if not raw:
        if split == "external":
            ck = fold_ckpts(paths)
            if not ck:
                raise SystemExit("[thr] --split external 需要折权重，但 checkpoints/ 下没有 "
                                 "g4_fold*/best.pth；请先训练或显式 --ckpt")
            raw = ",".join(ck)
        else:
            raw = os.path.join(paths["checkpoints_dir"], f"g4_fold{a.fold}", "best.pth")
    ckpts: list[str] = []
    for c in str(raw).split(","):
        c = c.strip()
        if c:
            ckpts.append(c if os.path.isabs(c) else resolve(c))
    for c in ckpts:
        if not os.path.exists(c):
            raise SystemExit(f"[thr] 权重不存在: {c}")

    # 标定集：external = 官方验证集（与训练集无交集，全折集成即无泄漏）；
    #         fold     = 折内 val（旧口径）。
    if split == "external":
        man, man_path = external_val_manifest()
        if man is None:
            raise SystemExit(
                f"[thr] --split external 但验证集清单不可用：{man_path}\n"
                f"  · 生成：export VAL_ROOT=<验证集目录> && bash scripts/01_probe.sh --val")
        n_img = sum(1 for c in man["cases"] if c.get("images"))
        val = [c for c in man["cases"] if c.get("images") and c.get("masks")]
        if n_img > len(val):
            print(f"[thr] ⚠️ {n_img - len(val)} 例无掩码，不能参与阈值标定")
        src = f"官方验证集（{man_path}）"
    else:
        with open(resolve(paths["folds"]), encoding="utf-8") as f:
            folds = json.load(f)
        with open(resolve(paths["manifest"]), encoding="utf-8") as f:
            man = json.load(f)
        by = {c["accession"]: c for c in man["cases"]}
        val = [by[x] for x in folds[str(a.fold)]["val"] if x in by]
        src = f"fold{a.fold} 折内 val"
    if a.limit:
        val = val[: a.limit]
    _tag = "（多折集成）" if len(ckpts) > 1 else ""
    print(f"[thr] {src} 验证 {len(val)} 例；权重 {len(ckpts)} 个{_tag}", flush=True)
    for _c in ckpts:
        print(f"        · {_c}", flush=True)

    pre = load_config("preprocess.yaml")
    pipe = GliomaPipeline(ckpts)
    cache_dir = resolve_cache_dir()
    grid = [float(x) for x in a.grid.split(",")]

    def _prep(c):
        try:
            return load_case_cached(c, pre, cache_dir), None
        except Exception as e:                                    # noqa: BLE001
            return None, e

    ex = ThreadPoolExecutor(max_workers=2)
    order = list(range(len(val)))
    futs = {i: ex.submit(_prep, val[i]) for i in order[:3]}
    nxt = 3
    probs, gts = [], []
    for i in order:
        while nxt < len(order) and len(futs) < 3:
            futs[nxt] = ex.submit(_prep, val[nxt])
            nxt += 1
        got, err = futs.pop(i).result()
        if err is not None or got is None:
            print(f"[thr] ✗ {val[i]['accession']}: {err}")
            continue
        vol, aff, masks = got
        try:
            res = pipe.predict_prob(val[i], vol)
        except Exception as e:                                    # noqa: BLE001
            print(f"[thr] ✗ {val[i]['accession']}: {e}")
            continue
        probs.append(res["seg"])
        gts.append(make_targets(masks, vol.shape[1:]))
    ex.shutdown(wait=False)
    if not probs:
        print("[thr] ✗ 没有可用样本")
        return 1

    best_thr, best_dice = [], []
    for c in range(2):
        bt, bd = 0.5, -1.0
        for t in grid:
            num = den = 0.0
            for p, g in zip(probs, gts):
                pr, gt = p[c] > t, g[c] > 0.5
                num += 2.0 * float((pr & gt).sum())
                den += float(pr.sum() + gt.sum())
            d = num / den if den > 0 else 1.0
            if d > bd:
                bt, bd = float(t), d
        best_thr.append(bt)
        best_dice.append(bd)
    print(f"[thr] 全图最优阈值 core={best_thr[0]:.2f}（Dice {best_dice[0]:.4f}） "
          f"peri={best_thr[1]:.2f}（Dice {best_dice[1]:.4f}）")
    print(f"[thr] 原阈值 {pipe.thresholds}")

    if a.write:
        # 多折集成时**写回所有折**：load_ensemble 对各折阈值取均值，
        # 只有各折一致，集成实际使用的阈值才等于此处标定出的最优值。
        for _c in ckpts:
            ck = torch.load(_c, map_location="cpu", weights_only=False)
            ck["thresholds"] = [best_thr[0], best_thr[1]]
            ck["thresholds_source"] = {"method": "full-volume-sweep",
                                       "n_val": len(probs), "n_ckpt": len(ckpts)}
            torch.save(ck, _c)
            print(f"[thr] 已写回 {_c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
