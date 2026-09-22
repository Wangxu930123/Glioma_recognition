#!/usr/bin/env python
"""BraTS2021 → 赛道四数据布局（**仅用于本地实验验证，不用于正式提交训练**）。

合规说明：正式提交的权重必须只用大赛提供的医学影像数据训练。本脚本把公开
BraTS 数据整理成与赛道四**完全一致**的目录/命名/字段结构，用于验证工程链路
（探针、掩码角色归位、1mm 公共网格、目标一二、重复影像、答案格式）与算法上限。

映射关系（BraTS 标签 → 赛道四任务）：
    ET  = label 4            → ``T1C/肿瘤瘤体`` = 任务A core（T1 增强核心区）
    TC  = label 1|4          → ``FLAIR/瘤体``（非水肿部分，参与 peri 并集）
    ED  = label 2            → ``FLAIR/水肿``
    WT  = label 1|2|4        → peri（= 瘤体 ∪ 水肿，验证并集逻辑）

用法：
    python scripts/10_brats_to_track4.py --src <BraTS2021_Training_Data> --dst <out> --n 500
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _save(arr, aff, path):
    import nibabel as nib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(arr, aff), path)


def convert_case(src_case: str, dst_case: str, name: str):
    import nibabel as nib

    def _load(mod):
        p = os.path.join(src_case, f"{name}_{mod}.nii.gz")
        if not os.path.isfile(p):
            return None, None
        img = nib.load(p)
        return np.asanyarray(img.dataobj).astype(np.float32), np.asarray(img.affine, float)

    seg_img = nib.load(os.path.join(src_case, f"{name}_seg.nii.gz"))
    seg = np.asanyarray(seg_img.dataobj).astype(np.uint8)
    aff = np.asarray(seg_img.affine, float)
    et = (seg == 4) | (seg == 3)                              # 增强肿瘤
    ed = seg == 2
    wt = (seg > 0)
    tc = et | (seg == 1)                                      # 瘤体（坏死核心 + 增强）
    n_vox = int(wt.sum())

    for mod, out_name in (("t1ce", "t1c"), ("t1", "t1"), ("t2", "t2"), ("flair", "flair")):
        arr, a = _load(mod)
        if arr is None:
            continue
        _save(arr, a, os.path.join(dst_case, f"{out_name}_0000", f"{out_name}.nii.gz"))

    _save(et.astype(np.uint8), aff, os.path.join(dst_case, "t1c_0000", "肿瘤瘤体.nii.gz"))
    _save(tc.astype(np.uint8), aff, os.path.join(dst_case, "flair_0000", "瘤体.nii.gz"))
    _save(ed.astype(np.uint8), aff, os.path.join(dst_case, "flair_0000", "水肿.nii.gz"))
    return n_vox


def make_fake(src_root: str, dst_root: str, names: list[str], n: int = 6, seed: int = 0):
    """假人体：破坏脑结构（颠倒/纯噪声/层错位），仅用于跑通目标一链路。"""
    import nibabel as nib
    rng = np.random.default_rng(seed)
    for k in range(n):
        name = names[k % len(names)]
        src = os.path.join(src_root, name)
        ref = nib.load(os.path.join(src, f"{name}_t1ce.nii.gz"))
        aff = np.asarray(ref.affine, float)
        acc = f"FAKE_{k:03d}"
        for mod_src, mod_out in (("t1ce", "t1c"), ("flair", "flair"), ("t2", "t2")):
            img = nib.load(os.path.join(src, f"{name}_{mod_src}.nii.gz"))
            a = np.asarray(img.dataobj).astype(np.float32)
            if k % 3 == 0:
                a = a[::-1]                                           # 上下颠倒
            elif k % 3 == 1:
                a = rng.standard_normal(a.shape).astype(np.float32) * 500   # 纯噪声
            else:
                a = np.roll(a, a.shape[2] // 3, axis=2)               # 层错位
            _save(a, np.asarray(img.affine, float),
                  os.path.join(dst_root, "annotation", "fake", acc,
                               f"{mod_out}_0000", f"{mod_out}.nii.gz"))


def make_composition(src_root: str, dst_root: str, names: list[str], n: int = 6):
    """拼接影像：病例 A 的上半 + 病例 B 的下半（层间不连续）。"""
    import nibabel as nib
    for k in range(n):
        na = names[(2 * k) % len(names)]
        nb = names[(2 * k + 1) % len(names)]
        acc = f"COMP_{k:03d}"
        for mod_src, mod_out in (("t1ce", "t1c"), ("flair", "flair"), ("t2", "t2")):
            ia = nib.load(os.path.join(src_root, na, f"{na}_{mod_src}.nii.gz"))
            ib = nib.load(os.path.join(src_root, nb, f"{nb}_{mod_src}.nii.gz"))
            a = np.asarray(ia.dataobj).astype(np.float32)
            b = np.asarray(ib.dataobj).astype(np.float32)
            m = a.shape[2] // 2
            c = np.concatenate([a[:, :, :m], b[:, :, m:]], axis=2)
            _save(c, np.asarray(ia.affine, float),
                  os.path.join(dst_root, "annotation", "Composition", acc,
                               f"{mod_out}_0000", f"{mod_out}.nii.gz"))


def make_duplicates(src_root: str, dst_root: str, names: list[str], n_pairs: int = 12,
                    seed: int = 1):
    """重复影像：``{ACC}_DUP`` = 原病例 + 轻微强度扰动（模拟同检查重复上传）。"""
    import nibabel as nib
    rng = np.random.default_rng(seed)
    pairs = []
    for k in range(min(n_pairs, len(names))):
        name = names[k]
        base = os.path.join(src_root, name)
        acc = f"{name}_DUP"
        for mod_out, mod_src in (("t1c", "t1ce"), ("t1", "t1"), ("t2", "t2")):
            img = nib.load(os.path.join(base, f"{name}_{mod_src}.nii.gz"))
            a = np.asarray(img.dataobj).astype(np.float32)
            a = a * (1.0 + 0.02 * rng.standard_normal(a.shape)).astype(np.float32)
            _save(a, np.asarray(img.affine, float),
                  os.path.join(dst_root, acc, f"{mod_out}_0000", f"{mod_out}.nii.gz"))
        seg_img = nib.load(os.path.join(base, f"{name}_seg.nii.gz"))
        seg = np.asanyarray(seg_img.dataobj)
        aff = np.asarray(seg_img.affine, float)
        masks = {"肿瘤瘤体": ((seg == 4) | (seg == 3)), "瘤体": ((seg == 4) | (seg == 3) | (seg == 1)),
                 "水肿": (seg == 2)}
        for role, m in masks.items():
            mod = "t1c" if role == "肿瘤瘤体" else "flair"
            _save(m.astype(np.uint8), aff,
                  os.path.join(dst_root, acc, f"{mod}_0000", f"{role}.nii.gz"))
        pairs.append((name, acc))
    d = os.path.join(dst_root, "annotation", "duplicate")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "gold.csv"), "w", encoding="utf-8") as f:
        f.write("src_img,desc_img\n")
        for a, b in pairs:
            f.write(f"{a},{b}\n")
    return pairs


def _synth_one(src_root: str, name: str):
    """单例合成结构化标签（模块级以便多进程 pickle）。"""
    import nibabel as nib
    import numpy as np
    from scipy import ndimage as ndi

    p_seg = os.path.join(src_root, name, f"{name}_seg.nii.gz")
    try:
        seg_img = nib.load(p_seg)
        seg = np.asanyarray(seg_img.dataobj)
        aff = np.asarray(seg_img.affine, float)
    except Exception:                                             # noqa: BLE001
        return None
    et = (seg == 4) | (seg == 3)
    ncr = seg == 1
    wt = seg > 0
    ed = seg == 2
    if not wt.any():
        return None
    v_et, v_ed = float(et.sum()), float(ed.sum())
    vol = float(wt.sum())
    er = ndi.binary_erosion(wt)
    surf = max(1.0, float(vol - er.sum()))
    r_eq = (3 * vol / (4 * np.pi)) ** (1 / 3)
    sph = float(np.clip((4 * np.pi * r_eq ** 2) / surf, 0, 1.5))

    ctr = np.argwhere(et if et.any() else wt).mean(0)
    nz = np.argwhere(wt)
    lo, hi = nz.min(0).astype(float), nz.max(0).astype(float)
    w = (aff[:3, :3] @ ctr) + aff[:3, 3]
    cw = (aff[:3, :3] @ np.array([lo, hi]).T).T + aff[:3, 3]
    n = (w - cw.min(0)) / np.maximum(cw.max(0) - cw.min(0), 1e-6)
    nx, ny, nz_ = float(n[0]), float(n[1]), float(n[2])
    side = "右侧" if nx >= 0.5 else "左侧"
    dx = abs(nx - 0.5) * 2
    if nz_ < 0.26:
        loc = "脑干" if (dx < 0.18 and ny < 0.60) else f"{side}小脑半球"
    elif dx < 0.30 and 0.30 <= nz_ < 0.62 and 0.25 <= ny <= 0.80:
        loc = f"{side}基底节区"
    elif ny < 0.28 and nz_ < 0.62:
        loc = f"{side}枕叶"
    elif dx > 0.42 and nz_ < 0.62:
        loc = f"{side}颞叶"
    elif ny > 0.58:
        loc = f"{side}额叶"
    elif nz_ >= 0.62:
        loc = f"{side}顶叶" if ny < 0.50 else f"{side}额叶"
    else:
        loc = f"{side}颞叶"

    grade = 4 if v_et > 3e4 else (3 if v_et > 1e4 else (2 if v_et > 3e3 else 1))
    ring, n_cc = 0.0, 0
    if et.any():
        ring = float(np.clip(1 - et.sum() / max(1.0, ndi.binary_fill_holes(et).sum()), 0, 1))
        n_cc = int(ndi.label(et)[1])
    if not et.any():
        pat = "无"
    elif n_cc >= 3:
        pat = "多灶状"
    elif ring > 0.45:
        pat = "环形"
    elif ring > 0.18:
        pat = "花环状"
    else:
        pat = "结节状"
    return {
        "AccessionNumber": name, "PatientId": name,
        "病理结果": f"脑胶质瘤{grade}级", "glioma_with_label": "是",
        "location_of_lesion": loc,
        "lesion_morphology": "规则" if sph > 0.55 else "不规则",
        "lesion_mor_feature_lobulation": "有/清" if sph <= 0.45 else "无/不清",
        "lesion_morph_feature_boundary": "有/清" if sph > 0.45 else "无/不清",
        "tumor_feature_necrosis": "有" if ncr.sum() > 200 else "无",
        "tumor_feature_change": "无", "tumor_feature_hemorrhage": "无",
        "tumor_feature_calcification": "无",
        "tumor_t2wi_signal_intensity": "3高" if v_ed > 0 else "2等",
        "tumor_t2_flair_sign_intensity": "3高" if v_ed > 0 else "2等",
        "tumor_t1wi_c_enhan": "有" if et.any() else "无",
        "tumor_t1wi_c_enhan_pattern": pat,
    }


def make_synth_labels(src_root: str, names: list[str], out_csv: str, workers: int = 8) -> int:
    """由 BraTS 掩码几何派生**合成结构化标签**（仅用于本地验证结构化头的训练链路）。

    ⚠️ 这些标签由影像特征规则生成，与推理侧规则部分同源，**不能**用其数值衡量
    真实比赛的结构化字段精度；它只用于验证字段映射、稀疏 label_mask、多任务头
    训练以及 prediction.json 的组装与枚举输出。
    """
    import csv
    import multiprocessing as mp

    rows = [r for r in mp.Pool(workers).starmap(_synth_one, [(src_root, n) for n in names],
                                                chunksize=4) if r]
    if not rows:
        return 0
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _worker(args):
    src, dst, name = args
    try:
        convert_case(src, dst, name)
        return None
    except Exception as e:                                        # noqa: BLE001
        return f"{name}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/data_sdb/wangx/data/Brain_MRI/BraTS2021_Training_Data")
    ap.add_argument("--dst", default="/mnt/data_sdb/wangx/data/Brain_MRI/track4_sim")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--synth-labels", action="store_true",
                    help="生成合成的结构化标签表（仅本地验证结构化头训练链路）")
    ap.add_argument("--special-only", action="store_true",
                    help="只补生成特殊影像与重复影像金标准（病例已转换时用）")
    a = ap.parse_args()

    names = sorted(os.listdir(a.src))
    names = [n for n in names if os.path.isdir(os.path.join(a.src, n))]
    random.Random(a.seed).shuffle(names)
    names = names[: a.n]
    os.makedirs(a.dst, exist_ok=True)

    if not a.special_only:
        print(f"[conv] {len(names)} 例 → {a.dst}（{a.workers} 进程）", flush=True)
        jobs = [(os.path.join(a.src, n), os.path.join(a.dst, n), n) for n in names]
        errs = []
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for i, e in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4)):
                if e:
                    errs.append(e)
                if (i + 1) % 50 == 0:
                    print(f"[conv] {i+1}/{len(names)} …", flush=True)
        print(f"[conv] 完成 {len(names) - len(errs)}/{len(names)}；失败 {len(errs)}")
        for e in errs[:5]:
            print(f"[conv] ✗ {e}")

    make_fake(a.src, a.dst, names, n=6, seed=a.seed)
    make_composition(a.src, a.dst, names, n=6)
    pairs = make_duplicates(a.src, a.dst, names, n_pairs=12, seed=a.seed)
    print(f"[conv] 特殊影像：fake 6 / Composition 6 / duplicate {len(pairs)} 对")

    if a.synth_labels:
        n = make_synth_labels(a.src, names, os.path.join(a.dst, "structured_synth.csv"),
                              workers=a.workers)
        print(f"[conv] 合成结构化标签 {n} 行 -> {a.dst}/structured_synth.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
