#!/usr/bin/env python
"""性能/显存基准：决定 patch 大小、batch、TTA 与集成折数前必须实测。

用法：
    python scripts/12_bench.py --root <赛道四格式数据根> [--patch 96] [--batch 2] [--folds 1]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DATASET_ROOT") or
                    "/mnt/data_sdb/wangx/data/Brain_MRI/track4_sim")
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--eval-n", type=int, default=2)
    ap.add_argument("--infer-patch", type=int, default=112)
    ap.add_argument("--folds", type=int, default=1)
    a = ap.parse_args()

    os.environ.setdefault("DATASET_ROOT", a.root)
    from src.data.dataset import GliomaDataset, build_case_volume
    from src.data.probe import scan_real
    from src.models.unet3d import build_model, cls_spec_from_config
    from src.utils.config import load_config
    from torch.utils.data import DataLoader

    pre = load_config("preprocess.yaml")
    cfg = load_config("train.yaml")
    fields = load_config("labels.yaml")["fields"]
    cases = [c for c in scan_real(a.root) if c.get("images")][:12]
    print(f"[bench] 可用病例 {len(cases)}（1mm 公共网格，patch={a.patch}³ batch={a.batch}）")
    if not cases:
        print("[bench] ✗ 无病例")
        return 1

    t0 = time.time()
    vol, aff, masks = build_case_volume(cases[0], pre)
    print(f"[bench] 单例加载+预处理 {time.time()-t0:.2f}s  vol{vol.shape} "
          f"spacing={[round(float(np.linalg.norm(aff[:3,i])),3) for i in range(3)]} "
          f"masks={ {k: int(v.sum()) for k, v in masks.items()} }")

    ds = GliomaDataset(cases, train=True, patch=(a.patch,) * 3, pre_cfg=pre,
                       label_fields=fields, aug_cfg=cfg, seed=0)
    t0 = time.time()
    _ = ds[0]
    print(f"[bench] 首个样本（含增广） {time.time()-t0:.2f}s")
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)

    model = build_model(cls_spec_from_config({"fields": fields}),
                        in_ch=len(pre["channels"]), arch=cfg["model"]["arch"],
                        base=cfg["model"]["base"], depth=cfg["model"]["depth"],
                        blocks_per_stage=cfg["model"]["blocks_per_stage"]).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[bench] 模型 {cfg['model']['arch']} 参数量 {n_par:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    t0 = time.time()
    it = iter(dl)
    for step in range(a.steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        x = b["image"].cuda().float()
        w = b["whole"].cuda().float()
        y = b["target"].cuda().float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out["seg"], y)
            if step % 4 == 0:
                gout = model(w)
                loss = loss + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(
                    gout["special"], torch.zeros_like(gout["special"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1:
            t0 = time.time()
    dt = (time.time() - t0) / max(1, a.steps - 1)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"[bench] 训练 {dt*1000:.0f} ms/step（含每 4 步一次全局头）  峰值显存 {peak:.2f} GB")
    used = peak
    if used > 22:
        print("[bench] ⚠️ 显存接近 24GB 上限，建议降 patch 或 batch")

    # 推理速度
    from src.inference.pipeline import GliomaPipeline
    from src.utils.config import load_paths
    ck = os.path.join(load_paths()["checkpoints_dir"], "_bench.pth")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.save({"model_ema": model.state_dict(), "cls_spec": cls_spec_from_config({"fields": fields}),
                "arch": cfg["model"]["arch"], "model_cfg": cfg["model"],
                "thresholds": [0.5, 0.5], "global_size": 96, "global_size_mm": 192,
                "special_trained": True}, ck)
    pre2 = dict(pre)
    pre2["inference"] = dict(pre["inference"])
    pre2["inference"]["patch"] = [a.infer_patch] * 3
    pipe = GliomaPipeline([ck])
    pipe.pre = pre2
    pipe.gcfg = {"out": 96, "size_mm": 192}
    t0 = time.time()
    r = pipe.process_case(cases[0])
    print(f"[bench] 单例推理 {time.time()-t0:.1f}s（patch={a.infer_patch}³, "
          f"TTA={pre['inference']['seg_tta_flips']}, 折数={a.folds}）"
          f"  → 100 例约 {(time.time()-t0)*100/60:.0f} 分钟")
    print(f"[bench] 输出：core={int(np.sum(r['masks']['core']))} vox, "
          f"peri={int(np.sum(r['masks']['peri']))} vox")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
