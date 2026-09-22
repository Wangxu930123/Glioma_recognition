#!/usr/bin/env python
"""预处理缓存：把 1mm 公共网格体积 + 掩码 + 整脑视图一次性落盘。

动机：训练时每个 epoch 每例只采样一次，内存 LRU 命中率极低，实测瓶颈在
"nii.gz 解码 + 重采样 + z-score"（约 2.2s/例）而非 GPU（约 0.5s/step）。
落盘为 float16（体积减半）+ uint8 掩码，训练时直接读数组，实测提速 3~5 倍。

用法：
    python scripts/13_build_cache.py [--root <数据根>] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _cache_paths(cache_dir: str, acc: str) -> dict:
    d = os.path.join(cache_dir, acc)
    return {"dir": d, "vol": os.path.join(d, "vol.npy"), "core": os.path.join(d, "core.npy"),
            "peri": os.path.join(d, "peri.npy"), "whole": os.path.join(d, "whole.npy"),
            "meta": os.path.join(d, "meta.json")}


def build_one(case: dict, cache_dir: str, gsize: int, size_mm: float, force: bool = False):
    from src.data.dataset import build_case_volume, global_view, lesion_center, make_targets

    cp = _cache_paths(cache_dir, case["accession"])
    if not force and os.path.isfile(cp["meta"]) and os.path.isfile(cp["vol"]):
        return "skip"
    from src.utils.config import load_config
    pre = load_config("preprocess.yaml")
    try:
        vol, aff, masks = build_case_volume(case, pre)
        tgt = make_targets(masks, vol.shape[1:])
        ctr = lesion_center(tgt)
        whole = global_view(vol, ctr, size_mm, gsize)
    except Exception as e:                                        # noqa: BLE001
        return f"{case['accession']}: {e}"
    os.makedirs(cp["dir"], exist_ok=True)
    np.save(cp["vol"], vol.astype(np.float16))
    np.save(cp["core"], (tgt[0] > 0).astype(np.uint8))
    np.save(cp["peri"], (tgt[1] > 0).astype(np.uint8))
    np.save(cp["whole"], whole.astype(np.float16))
    with open(cp["meta"], "w", encoding="utf-8") as f:
        json.dump({"accession": case["accession"], "shape": list(vol.shape),
                   "affine": np.asarray(aff, float).tolist(), "whole": list(whole.shape),
                   # 病灶质心：训练侧可据此直接切片读取（避免全量载入 71MB/例）
                   "lesion_center": (np.asarray(ctr, float).tolist() if ctr is not None else None)}, f)
    return None


def _worker(arg):
    case, cache_dir, gsize, size_mm, force = arg
    return build_one(case, cache_dir, gsize, size_mm, force)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DATASET_ROOT") or
                    "/mnt/data_sdb/wangx/data/Brain_MRI/track4_sim")
    ap.add_argument("--cache", default=os.environ.get("CACHE_DIR") or "data/preprocess_cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("DATASET_ROOT", a.root)

    from src.data.probe import scan_real
    from src.utils.config import load_config, resolve
    cfg = load_config("train.yaml")
    gv = cfg.get("global_view") or {}
    cases = scan_real(a.root)
    cache_dir = resolve(a.cache)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[cache] {len(cases)} 例 → {cache_dir}（{a.workers} 进程）", flush=True)

    import multiprocessing as mp
    jobs = [(c, cache_dir, int(gv.get("out", 96)), float(gv.get("size_mm", 192)), a.force)
            for c in cases]
    errs, skips, done = [], 0, 0
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=2)):
            if r == "skip":
                skips += 1
            elif r:
                errs.append(r)
            else:
                done += 1
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"[cache] {i+1}/{len(jobs)}  已建 {done} 跳过 {skips} 失败 {len(errs)}  "
                      f"({el:.0f}s, {el/(i+1):.2f}s/例)", flush=True)
    print(f"[cache] 完成：新建 {done}，跳过 {skips}，失败 {len(errs)}")
    for e in errs[:5]:
        print(f"[cache] ✗ {e}")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
