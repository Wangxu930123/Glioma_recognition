#!/usr/bin/env python
"""合成数据端到端自检（无需真实数据）：证明"工程能跑通 + 答案格式完全合规"。

覆盖：
 1. 造 6 例合成 MRI（t1c/flair/t2 + T1C 核心区掩码、FLAIR 总异常区掩码）
    + ``annotation/fake``（假人体）、``annotation/Composition``（拼接）
    + ``annotation/duplicate/gold.csv`` + 结构化金标准 csv；
 2. 探针 → manifest（含特殊影像病例合并）→ 折划分；
 3. 训练 2 epoch（极小尺寸，验证**分割 + 结构化 + 特殊影像 + 重复嵌入**四路损失
    与规范 JSONL 日志）；
 4. 对"只含影像"的测评目录跑推理流水线 → 写答案；
 5. 强制规范校验（0/1、仿射/维度一致、duplicate 字段、≤200 对/例、UID 属于测试列表）；
 6. 打印 PASS/FAIL。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

WORK = os.environ.get("SMOKE_DIR") or os.path.join(tempfile.gettempdir(), "glioma4_smoke")


# --------------------------------------------------------------------------- #
def _syn_case(cdir: str, shape=(32, 32, 24), spacing=(1.0, 1.0, 2.0), seed=0,
              with_masks: bool = True, fake: bool = False):
    import nibabel as nib
    rng = np.random.default_rng(seed)
    aff = np.diag(list(spacing) + [1.0])
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    ctr = np.array([shape[0] // 2 + rng.integers(-2, 3), shape[1] // 2 + rng.integers(-2, 3),
                    shape[2] // 2])
    r_core, r_peri = 4 + seed % 3, 7 + seed % 3
    d_core = ((zz - ctr[0]) ** 2 + (yy - ctr[1]) ** 2 + (xx - ctr[2]) ** 2) <= r_core ** 2
    d_peri = ((zz - ctr[0]) ** 2 + (yy - ctr[1]) ** 2 + (xx - ctr[2]) ** 2) <= r_peri ** 2
    brain = (0.3 + 0.01 * rng.standard_normal(shape)).astype(np.float32)
    if fake:                                                       # 假人体：脑外高信号条纹
        brain = np.zeros(shape, np.float32) + 0.02 * rng.standard_normal(shape)
        brain[::3] += 3.0
    for mod, boost in (("t1c", 1.6), ("flair", 1.2), ("t2", 1.1)):
        vol = brain.copy()
        vol[d_peri] += boost
        vol[d_core] += 0.8
        mdir = os.path.join(cdir, f"{mod}_0000")
        os.makedirs(mdir, exist_ok=True)
        nib.save(nib.Nifti1Image(vol.astype(np.float32), aff), os.path.join(mdir, f"{mod}.nii.gz"))
    if with_masks:
        # 掩码命名与真实标注一致：T1C → "肿瘤瘤体"；FLAIR → "瘤体"（总异常区组成）
        nib.save(nib.Nifti1Image(d_core.astype(np.uint8), aff),
                 os.path.join(cdir, "t1c_0000", "肿瘤瘤体.nii.gz"))
        nib.save(nib.Nifti1Image(d_peri.astype(np.uint8), aff),
                 os.path.join(cdir, "flair_0000", "瘤体.nii.gz"))
        nib.save(nib.Nifti1Image((d_peri & ~d_core).astype(np.uint8), aff),
                 os.path.join(cdir, "flair_0000", "水肿.nii.gz"))
    return aff


def make_synthetic(root: str, n: int = 6) -> dict:
    import csv

    os.makedirs(root, exist_ok=True)
    accs = []
    for i in range(n):
        acc = f"GLIOMA_{i:03d}"
        accs.append(acc)
        _syn_case(os.path.join(root, acc), seed=i, with_masks=True)

    # 结构化金标准
    with open(os.path.join(root, "structured.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["AccessionNumber", "PatientId", "病理结果", "location_of_lesion",
                    "lesion_morphology", "tumor_t1wi_c_enhan", "tumor_t1wi_c_enhan_pattern",
                    "tumor_feature_necrosis", "tumor_feature_change", "tumor_feature_hemorrhage",
                    "tumor_feature_calcification", "lesion_morph_feature_boundary",
                    "lesion_mor_feature_lobulation", "tumor_t2wi_signal_intensity",
                    "tumor_t2_flair_sign_intensity"])
        for i, acc in enumerate(accs):
            w.writerow([acc, f"P{i:04d}", "脑胶质瘤4级" if i % 3 == 0 else "脑胶质瘤2级",
                        "右侧额叶" if i % 2 == 0 else "左侧颞叶", "不规则",
                        "有", "花环状", "有", "无", "无", "无", "无/不清", "有/清", "3高", "3高"])

    # 目标一/二：特殊影像（正样本内含影像，可被探针合并）
    for cls, ids in (("fake", ["FAKE_001", "FAKE_002"]),
                     ("Composition", ["COMP_001", "COMP_002"])):
        for k, ident in enumerate(ids):
            _syn_case(os.path.join(root, "annotation", cls, ident), seed=100 + k,
                      with_masks=False, fake=(cls == "fake"))

    # 重复影像金标准（第 0 与第 1 例重复）
    du = os.path.join(root, "annotation", "duplicate")
    os.makedirs(du, exist_ok=True)
    with open(os.path.join(du, "gold.csv"), "w", encoding="utf-8") as f:
        f.write("src_img,desc_img\n")
        f.write(f"{accs[0]},{accs[1]}\n")
    return {"root": root, "accessions": accs}


def make_eval_only(src_root: str, dst_root: str, accessions: list[str]) -> None:
    """构造"只含影像"的测评目录（无掩码、无金标准）。"""
    os.makedirs(dst_root, exist_ok=True)
    for acc in accessions:
        for series in ("t1c_0000", "flair_0000", "t2_0000"):
            s = os.path.join(src_root, acc, series)
            d = os.path.join(dst_root, acc, series)
            if os.path.isdir(s):
                os.makedirs(d, exist_ok=True)
                for fn in os.listdir(s):
                    if fn.endswith((".nii", ".nii.gz")):
                        shutil.copy2(os.path.join(s, fn), os.path.join(d, fn))


# --------------------------------------------------------------------------- #
def main() -> int:
    os.environ.setdefault("WORKSPACE", os.path.join(WORK, "workspace"))
    os.environ["DATASET_ROOT"] = os.path.join(WORK, "train")
    os.environ["ANSWER_ROOT"] = os.path.join(WORK, "answer")
    os.environ["LOGS_DIR"] = os.path.join(WORK, "workspace", "logs")
    os.environ["DICOM_CACHE"] = os.path.join(WORK, "cache")
    # **隔离**：清单/折文件必须写在临时目录，否则会覆盖真实实验的 data/manifest.json
    os.environ["MANIFEST"] = os.path.join(WORK, "manifest.json")
    os.environ["FOLDS"] = os.path.join(WORK, "folds.json")
    os.environ["CKPT_DIR"] = os.path.join(WORK, "checkpoints")
    os.environ["CACHE_DIR"] = os.path.join(WORK, "preprocess_cache")
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(os.environ["LOGS_DIR"], exist_ok=True)
    os.makedirs(os.environ["CKPT_DIR"], exist_ok=True)

    print("[smoke] 1/6 造合成数据（含 fake / Composition / duplicate）…")
    syn = make_synthetic(os.environ["DATASET_ROOT"], n=6)
    evl = os.path.join(WORK, "eval")
    make_eval_only(syn["root"], evl, syn["accessions"][-2:])

    print("[smoke] 2/6 探针 + 折划分…")
    from src.data.dataset import build_folds
    from src.data.probe import probe
    from src.utils.config import load_paths, resolve
    res = probe(syn["root"])
    man_p = resolve(load_paths()["manifest"])
    os.makedirs(os.path.dirname(man_p), exist_ok=True)
    with open(man_p, "w", encoding="utf-8") as f:
        json.dump({"cases": res["cases"], "special": res["special"], "report": res["report"]},
                  f, ensure_ascii=False, indent=1)
    rep = res["report"]
    print(f"        病例 {rep['n_cases']} | 模态 {rep['modality_counts']} | 掩码角色 {rep['mask_role_counts']}")
    print(f"        结构化字段 {rep['label_field_counts']} | 重复金标准 "
          f"{(res['special'].get('gold_pairs') or [])[:2]}")
    assert rep["n_cases"] >= 6, "探针未识别出病例"
    assert rep["modality_counts"].get("t1c", 0) >= 6, "未识别 T1C 序列"
    assert rep["mask_role_counts"].get("core", 0) >= 6, "未识别核心区掩码"
    assert rep["mask_role_counts"].get("peri", 0) >= 6, "未识别总异常区掩码"
    assert rep["label_field_counts"], "未读到结构化金标准"
    assert len(res["special"]["fake_cases"]) >= 2 and len(res["special"]["composition_cases"]) >= 2
    assert res["special"]["gold_pairs"], "未读到重复影像金标准"
    build_folds(load_paths()["manifest"], n_folds=2, val_ratio=0.34)

    print("[smoke] 3/6 训练 2 epoch（分割 + 结构化 + 特殊 + 嵌入 四路损失）…")
    import yaml

    from src.utils.config import PROJECT_ROOT, load_yaml
    smoke_cfg = {
        "seed": 0, "epochs": 2, "batch_size": 1, "patch_size": [32, 32, 16],
        "lr": 1e-3, "weight_decay": 0.0, "amp_dtype": "bfloat16", "ema_decay": 0.9,
        "grad_clip": 12.0, "num_workers": 0, "val_every": 1, "folds": 2, "val_ratio": 0.34,
        "threshold_grid": [0.3, 0.5, 0.7],
        "checkpoint_metric": "val_dice_peri",
        "model": {"arch": "mednext", "base": 8, "depth": 3, "blocks_per_stage": 1, "k": 3,
                  "expand": 2, "dropout": 0.0, "aniso_z": False},
        "sampler": {"pos_ratio": 0.8},
        "augment": {"enabled": True, "flip_prob": 0.5, "elastic_prob": 0.0, "scale_prob": 0.0},
        "global_view": {"size_mm": 64, "out": 32, "every": 1},
        "aux": {"special_batch": 2, "pair_batch": 2, "neg_per_pos": 2},
        "loss": {"dice": 1.0, "ce": 1.0, "deep_sup": 0.4, "cls": 1.0, "special": 1.0,
                 "embed": 0.3, "contain": 0.2, "boundary": 0.0, "seg_pos_weight": 1.0,
                 "cls_pos_weight": 2.0, "embed_scale": 10.0},
    }
    cfg_p = os.path.join(PROJECT_ROOT, "configs", "_smoke_train.yaml")
    with open(cfg_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(smoke_cfg, f, allow_unicode=True)
    from src.training.trainer import train
    ckpt = train("_smoke_train", 0, "smoke_fold0", no_resume=True)
    assert os.path.exists(ckpt), "训练未产出 checkpoint"
    print(f"        checkpoint -> {ckpt}")

    print("[smoke] 4/6 推理流水线（只含影像的测评目录）…")
    os.environ["GLIOMA_CKPT"] = ckpt
    from src.inference.pipeline import GliomaPipeline
    pipe = GliomaPipeline([ckpt])
    out_dir = os.path.join(WORK, "answer", "eval_0001")
    r = pipe.run_batch(evl, out_dir, gold_pairs=[syn["accessions"][-2:]])
    print(f"        完成 {r['n_done']}/{r['n_cases']} 例，重复对 {r['pairs']}")

    print("[smoke] 5/6 强制规范校验…")
    from src.inference.writer import validate_answer
    v = validate_answer(out_dir, known_accessions={c["accession"] for c in res["cases"]})
    print(f"        ok={v['ok']} n_cases={v['n_cases']} errors={v['errors'][:3]}")

    print("[smoke] 6/6 检查答案目录与 prediction.json 字段…")
    acc0 = syn["accessions"][-2]
    pj = os.path.join(out_dir, acc0, "prediction.json")
    with open(pj, encoding="utf-8") as f:
        d = json.load(f)
    need_top = ["AccessionNumber", "IsNotHumanBodyProb", "IsStitchedProb", "ProcessingTime_ms",
                "SegmentationMaskURI", "Prediction", "Interpretation"]
    miss = [k for k in need_top if k not in d]
    need_pred = ["TumorProbability", "Location", "Morphology", "WHO_Grade", "Enhancement",
                 "EnhancementPattern", "Necrosis", "CysticChange", "Hemorrhage", "Calcification",
                 "Margin", "Lobulation", "Signal_T2WI", "Signal_FLAIR"]
    miss_p = [k for k in need_pred if k not in d.get("Prediction", {})]
    dp = os.path.join(out_dir, "duplicate_pairs.jsonl")
    n_pairs = sum(1 for _ in open(dp, encoding="utf-8")) if os.path.exists(dp) else 0
    # 掩码几何复核（写盘时已回读校验，此处再独立确认）
    import nibabel as nib
    geo_ok = True
    for key, rel in (d.get("SegmentationMaskURI") or {}).items():
        p = os.path.normpath(os.path.join(os.path.dirname(pj), str(rel)))
        img = nib.load(p)
        geo_ok &= set(np.unique(np.asanyarray(img.dataobj)).tolist()) <= {0, 1}

    ok = (not miss) and (not miss_p) and v["ok"] and n_pairs > 0 and geo_ok
    print(f"        顶层字段缺失={miss} | Prediction 字段缺失={miss_p} | "
          f"duplicate 行数={n_pairs} | 掩码 0/1 合规={geo_ok}")
    print("=" * 62)
    print(f"[smoke] {'PASS ✔ 全链路与答案格式均合规' if ok else 'FAIL ✗ 见上方错误'}")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
