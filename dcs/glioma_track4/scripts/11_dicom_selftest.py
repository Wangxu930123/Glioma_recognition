#!/usr/bin/env python
"""DICOM 兼容性自检：证明"测试集是 DICOM 时工程仍能产出合规答案"。

《赛事开发规范（赛道四）》原文：*影像原始数据为dicom格式*；
《公共数据集格式说明》：*影像原始数据为 nii 格式*。两者冲突，因此必须以
**两种格式都能工作**为准——本脚本把 NIfTI 转成真实 DICOM 序列，然后：
  1. 探针扫描 → 检查序列/模态识别；
  2. 推理流水线 → 检查掩码 affine/shape 与 DICOM 恢复出的几何**逐字节一致**；
  3. 答案格式强制校验。

用法：
    python scripts/11_dicom_selftest.py --src <赛道四格式数据根> --n 2 [--ckpt <权重>]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def nifti_to_dicom(src_nii: str, out_dir: str, series_uid: str, study_uid: str,
                   series_desc: str) -> bool:
    """把一个 NIfTI 序列写成 DICOM 多文件序列（SimpleITK）。"""
    import SimpleITK as sitk

    os.makedirs(out_dir, exist_ok=True)
    img = sitk.ReadImage(src_nii)
    img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkInt16)   # GDCM 不支持浮点直接写
    size = img.GetSize()
    if size[2] < 3:
        return False
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for k in range(size[2]):
        sl = img[:, :, k]
        sl.SetMetaData("0008|0060", "MR")
        sl.SetMetaData("0008|103e", series_desc)              # SeriesDescription
        sl.SetMetaData("0020|000d", study_uid)                # StudyInstanceUID
        sl.SetMetaData("0020|000e", series_uid)               # SeriesInstanceUID
        sl.SetMetaData("0020|0011", "1")                      # SeriesNumber
        sl.SetMetaData("0020|0013", str(k + 1))               # InstanceNumber
        writer.SetFileName(os.path.join(out_dir, f"{k:04d}.dcm"))
        writer.Execute(sl)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/data_sdb/wangx/data/Brain_MRI/track4_sim")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "glioma4_dicom"))
    a = ap.parse_args()

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True)
    os.environ["DICOM_CACHE"] = os.path.join(a.out, "cache")
    os.environ.setdefault("LOGS_DIR", os.path.join(a.out, "logs"))

    cases = sorted(d for d in os.listdir(a.src)
                   if os.path.isdir(os.path.join(a.src, d)) and d.startswith("BraTS"))
    cases = cases[: a.n]
    if not cases:
        print(f"[dicom] ✗ {a.src} 下没有可用病例（先跑 scripts/10_brats_to_track4.py）")
        return 1

    ds = os.path.join(a.out, "dataset")
    src_geo = {}
    print(f"[dicom] 1/4 生成 DICOM 测评集（{len(cases)} 例）…")
    for acc in cases:
        for mod in ("t1c", "flair", "t2", "t1"):
            p = os.path.join(a.src, acc, f"{mod}_0000", f"{mod}.nii.gz")
            if not os.path.isfile(p):
                continue
            uid = f"1.2.826.0.1.3680043.10.{abs(hash(acc + mod)) % 10 ** 9}"
            ok = nifti_to_dicom(p, os.path.join(ds, acc, f"{uid}"),
                                uid, f"1.2.826.0.1.3680043.10.99.{abs(hash(acc)) % 10 ** 8}",
                                f"AX {mod.upper()} TSE")
            if ok and mod in ("t1c", "flair"):
                import nibabel as nib
                img = nib.load(p)
                src_geo.setdefault(acc, {})[mod] = (tuple(img.shape),
                                                    np.asarray(img.affine, float))
    print(f"        源几何记录 {len(src_geo)} 例")

    print("[dicom] 2/4 探针扫描 DICOM …")
    from src.data.probe import scan_real
    found = scan_real(ds)
    by = {c["accession"]: c for c in found}
    for acc in cases:
        mods = sorted((by.get(acc, {}).get("images") or {}).keys())
        print(f"        {acc}: 模态={mods}")
    missing = [c for c in cases if c not in by or not by[c]["images"]]
    if missing:
        print(f"[dicom] ✗ 探针未识别 DICOM 病例: {missing}")
        return 1

    if not a.ckpt:
        print("[dicom] 3/4 跳过推理（未提供 --ckpt）。探针与几何解析通过 ✔")
        return 0

    print("[dicom] 3/4 推理 + 掩码几何比对 …")
    from src.inference.pipeline import GliomaPipeline
    from src.inference.writer import validate_answer
    pipe = GliomaPipeline([a.ckpt])
    out_dir = os.path.join(a.out, "answer", "eval_dicom")
    rep = pipe.run_batch(ds, out_dir)

    import nibabel as nib
    ok_geo = True
    for acc, want in src_geo.items():
        for role, mod in (("core", "t1c"), ("flair", "flair")):
            gp = os.path.join(out_dir, acc)
            if not os.path.isdir(gp):
                continue
            pj = os.path.join(gp, "prediction.json")
            if not os.path.isfile(pj):
                continue
            with open(pj, encoding="utf-8") as f:
                d = json.load(f)
            uri = (d.get("SegmentationMaskURI") or {}).get(role)
            if not uri:
                continue
            mp = os.path.normpath(os.path.join(gp, str(uri)))
            img = nib.load(mp)
            wshape, waff = want[mod]
            same = (tuple(img.shape) == wshape) and np.allclose(img.affine, waff, atol=1e-3)
            ok_geo &= same
            print(f"        {acc}/{role}: DICOM 空间一致={same}  (shape {img.shape} vs {wshape})")

    print("[dicom] 4/4 答案强制校验 …")
    v = validate_answer(out_dir, known_accessions={c["accession"] for c in found})
    ok = v["ok"] and ok_geo
    print("=" * 62)
    print(f"[dicom] {'PASS ✔ DICOM 路径与答案格式均合规' if ok else 'FAIL ✗ 见上方'}")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
