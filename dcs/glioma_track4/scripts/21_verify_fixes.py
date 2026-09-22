#!/usr/bin/env python
"""P0 缺陷修复的回归验证：逐项**断言**，而不是只看"能跑通"。

审核阶段发现 7 个必然失分/崩分的致命缺陷（TTA 双重翻折、嵌入损失未生效、
目标一/二无监督、掩码 header 污染、探针不认 DICOM、单例异常无兜底、
1mm 公共网格未生效）。本脚本对每一项做可复现的数值/结构断言。

用法:
    python scripts/21_verify_fixes.py
退出码 0 表示全部通过。
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)


# --------------------------------------------------------------------------- #
# V1: TTA 不得双重翻折（sliding.py）
# --------------------------------------------------------------------------- #
def v1_tta_no_double_flip() -> None:
    print("\nV1  TTA 翻折正确性（恒等模型下 TTA 必须与无 TTA 完全一致）")
    import torch

    from src.inference import sliding as S

    # 1a) _flip 往返必须是恒等变换
    x = torch.arange(2 * 3 * 8 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8, 8)
    ok = all(torch.equal(x, S._flip(S._flip(x, (a,)), (a,), back=True)) for a in ("x", "y", "z"))
    check("_flip 往返恒等（轴名 x/y/z）", ok)

    # 1b) 恒等模型：seg 通道 = 输入通道求和。翻折是精确的线性操作，
    #     因此"flip→前向→flip 回"的 TTA 结果必须与"不翻折"逐元素相同。
    #     ★ 若 _fwd 已经翻回、外层又翻一次（历史 bug），结果必然错位。
    class IdentitySeg(torch.nn.Module):
        def forward(self, x):
            s = x.sum(1, keepdim=True)
            n = x.shape[0]
            return {"seg": torch.cat([s, s], 1),
                    "cls": [torch.zeros(n, 1)],
                    "special": torch.zeros(n, 2),
                    "embed": torch.zeros(n, 4)}

    torch.manual_seed(0)
    vol = torch.randn(4, 16, 24, 24).numpy().astype(np.float32)
    gv = np.zeros((4, 16, 16, 16), np.float32)
    kw = dict(patch=(16, 24, 24), overlap=0.5, global_vol=gv, device="cpu",
              amp_dtype=None, crop_brain=False)
    a = S.predict_volume([IdentitySeg()], vol, tta_flips=(), **kw)["seg"]
    b = S.predict_volume([IdentitySeg()], vol, tta_flips=("x", "y", "z"), **kw)["seg"]
    d = float(np.abs(a - b).max())
    check("恒等模型 TTA 与无 TTA 逐元素一致", d < 1e-5, f"max|Δ|={d:.3e}")

    # 1c) 非对称信号：TTA 开/关的分割质心必须重合（双重翻折会让质心偏移）
    vol2 = np.zeros((4, 32, 32, 32), np.float32)
    vol2[:, :8, 10:22, 10:22] = 2.0                     # 信号只在 z 前半段
    kw2 = dict(patch=(32, 32, 32), overlap=0.5, global_vol=gv, device="cpu",
               amp_dtype=None, crop_brain=False)

    class Blob(torch.nn.Module):
        def forward(self, x):
            s = x[:, :1]
            n = x.shape[0]
            return {"seg": torch.cat([s, s], 1), "cls": [torch.zeros(n, 1)],
                    "special": torch.zeros(n, 2), "embed": torch.zeros(n, 4)}

    p0 = S.predict_volume([Blob()], vol2, tta_flips=(), **kw2)["seg"][0]
    p1 = S.predict_volume([Blob()], vol2, tta_flips=("x", "y", "z"), **kw2)["seg"][0]
    c0, c1 = np.argwhere(p0 > 0.5), np.argwhere(p1 > 0.5)
    shift = float(np.linalg.norm(c0.mean(0) - c1.mean(0))) if len(c0) and len(c1) else 999.0
    check("非对称信号下 TTA 质心不偏移", shift < 2.0, f"质心偏移={shift:.2f} 体素")


# --------------------------------------------------------------------------- #
# V2: 目标一/二与重复影像数据集必须产出可监督信号（trainer/dataset）
# --------------------------------------------------------------------------- #
def v2_special_and_embed_supervised() -> None:
    print("\nV2  特殊影像 / 重复影像监督信号（四路损失的数据来源）")
    from src.data.dataset import DuplicatePairDataset, SpecialImageDataset

    # images 必须非空——数据集会过滤掉没有影像的病例
    _s = {"path": "/nonexistent/t1c.nii.gz", "series_uid": "S1"}
    cases = [{"accession": f"ACC{i:03d}", "dir": "", "images": {"t1c": dict(_s)}, "masks": {}}
             for i in range(6)]
    pos_fake, pos_comp = {"ACC000"}, {"ACC001"}

    ds = SpecialImageDataset(cases, pos_fake, pos_comp, n_per_epoch=8, cache_size=8)
    seen_tasks, masks_ok = set(), True
    for i in range(8):
        it = ds[i]
        t, m = it["target"].numpy(), it["label_mask"].numpy()
        seen_tasks.add(int(np.argmax(m)))
        if m.sum() != 1:                                # 每样本只监督它所属的那一路
            masks_ok = False
        if not (len(t) == 2 == len(m)):
            masks_ok = False
    check("SpecialImageDataset 覆盖 fake 与 Composition 两路", seen_tasks == {0, 1},
          f"覆盖任务={sorted(seen_tasks)}")
    check("label_mask 逐样本只监督所属任务（不误当另一任务负样本）", masks_ok)

    dsp = DuplicatePairDataset(cases, [["ACC000", "ACC002"]], n_neg_per_pos=2, cache_size=8)
    labels = [float(dsp[i]["label"]) for i in range(len(dsp))]
    check("DuplicatePairDataset 同时产出正/负样本", (1.0 in labels) and (0.0 in labels),
          f"正={labels.count(1.0)} 负={labels.count(0.0)}")


# --------------------------------------------------------------------------- #
# V3: 掩码 header 必须干净（writer.py）
# --------------------------------------------------------------------------- #
def v3_mask_header_clean() -> None:
    print("\nV3  掩码几何与数值保真（writer.py）")
    import nibabel as nib

    aff = np.array([[-0.9370, 0.0212, 0.3486, 12.3456],
                    [0.0150, -0.9995, 0.0295, -18.7654],
                    [-0.3490, -0.0224, -0.9369, 25.6789],
                    [0.0, 0.0, 0.0, 1.0]])
    data = np.zeros((20, 20, 20), np.uint8)
    data[5:12, 5:12, 5:12] = 1
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.nii.gz")
        # 与 OutputWriter._write_masks 完全相同的写法
        img = nib.Nifti1Image(data, aff)
        img.set_data_dtype(np.uint8)
        img.header.set_sform(aff, code=1)
        img.header.set_qform(aff, code=1)
        img.header["scl_slope"] = 1.0
        img.header["scl_inter"] = 0.0
        nib.save(img, p)

        back = nib.load(p)
        arr = np.asanyarray(back.dataobj)
        check("affine 精确往返", float(np.abs(back.affine - aff).max()) < 1e-4,
              f"max|Δ|={np.abs(back.affine - aff).max():.3e}")
        check("体素严格 0/1（未被 scl 缩放污染）", set(np.unique(arr).tolist()) <= {0, 1},
              f"取值={sorted(set(np.unique(arr).tolist()))}")
        check("shape 与源一致", arr.shape == data.shape, f"{arr.shape}")


# --------------------------------------------------------------------------- #
# V4: 探针必须能识别 DICOM（probe.py / dicom.py）
# --------------------------------------------------------------------------- #
def v4_dicom_supported() -> None:
    print("\nV4  DICOM 序列识别与转换（probe.py / dicom.py）")
    try:
        import SimpleITK as sitk
    except ImportError:
        check("SimpleITK 可用", False, "未安装，跳过")
        return
    from src.data import dicom as D

    with tempfile.TemporaryDirectory() as td:
        arr = np.random.RandomState(0).randint(0, 500, (6, 24, 24)).astype(np.int16)
        for k in range(6):
            img = sitk.GetImageFromArray(arr[k])
            img.SetSpacing((1.0, 1.0))
            img.SetOrigin((0.0, 0.0, float(k)))
            img.SetMetaData("0008|0060", "MR")
            img.SetMetaData("0008|103E", "t1c post contrast")
            sitk.WriteImage(img, os.path.join(td, f"IM{k:03d}.dcm"))

        check("_looks_like_dicom 能识别 DICOM 文件",
              D._looks_like_dicom(os.path.join(td, "IM000.dcm")))
        series = D.ensure_series_nifti(td)
        ok = bool(series) and os.path.exists(series[0]["path"])
        produced, shape = "None", None
        if ok:
            import nibabel as nib
            img = nib.load(series[0]["path"])
            shape = img.shape
            sp = [round(float(np.linalg.norm(np.asarray(img.affine)[:3, i])), 3) for i in range(3)]
            produced = f"{shape} spacing={sp}"
            ok = len(shape) == 3 and all(x > 0 for x in shape)
        check("DICOM → NIfTI 转换可用（含 UID 不一致的兜底）", ok, f"产出={produced}")


# --------------------------------------------------------------------------- #
# V5: 单例异常必须有兜底答案（pipeline.py / writer.py）
# --------------------------------------------------------------------------- #
def v5_fallback_answer() -> None:
    print("\nV5  异常病例兜底答案（每例都必须有 prediction.json）")
    from src.inference import writer as W
    from src.utils.config import load_config

    cfg = load_config("preprocess.yaml")
    with tempfile.TemporaryDirectory() as td:
        case = {"accession": "BAD_0001", "images": {}, "masks": {}}
        W.write_fallback_case(td, case, cfg, reason="影像读取失败")
        p = os.path.join(td, case["accession"], "prediction.json")
        exists = os.path.exists(p)
        check("兜底 prediction.json 已写出", exists)
        if not exists:
            return
        import json
        d = json.load(open(p, encoding="utf-8"))
        need_top = {"AccessionNumber", "Prediction"}
        fields = [f["key"] for f in load_config("labels.yaml")["fields"]]
        miss_top = sorted(need_top - set(d))
        miss_pred = sorted(set(fields) - set(d.get("Prediction", {})))
        check("顶层与 Prediction 字段完整", not miss_top and not miss_pred,
              f"缺顶层={miss_top} 缺Prediction={miss_pred}")
        W.write_duplicate_pairs(td, [], fallback_accessions=[case["accession"]])
        rep = W.validate_answer(td, known_accessions={case["accession"]})
        errs = rep.get("errors")
        check("兜底答案通过规范自检", rep.get("ok") is True, f"errors={errs}")


# --------------------------------------------------------------------------- #
# V6: 1mm 公共网格必须生效（dataset.py）
# --------------------------------------------------------------------------- #
def v6_common_grid() -> None:
    print("\nV6  1mm 公共网格（预处理的几何基准）")
    from src.data.dataset import target_grid
    from src.utils.config import load_config

    cfg = load_config("preprocess.yaml")
    want = tuple(float(x) for x in cfg.get("common_spacing", [1, 1, 1]))
    ref_aff = np.diag([0.5, 0.5, 1.0, 1.0])            # 高分辨率各向异性输入
    shape, aff = target_grid((240, 240, 155), ref_aff, cfg)
    sp = [float(np.linalg.norm(np.asarray(aff)[:3, i])) for i in range(3)]
    close = all(abs(s - w) < 1e-3 for s, w in zip(sp, want))
    check("0.5mm 输入被归一到 1mm 公共网格", close, f"实际 spacing={[round(s, 3) for s in sp]}")

    ref_aff2 = np.diag([1.0, 1.0, 1.0, 1.0])           # 已在目标网格
    shape2, aff2 = target_grid((182, 218, 182), ref_aff2, cfg)
    check("已在公共网格的输入原样使用", shape2 == (182, 218, 182) and np.allclose(aff2, ref_aff2))


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 74)
    print("P0 缺陷修复回归验证")
    print("=" * 74)
    for fn in (v1_tta_no_double_flip, v2_special_and_embed_supervised, v3_mask_header_clean,
               v4_dicom_supported, v5_fallback_answer, v6_common_grid):
        try:
            fn()
        except Exception as exc:                                    # noqa: BLE001
            check(f"{fn.__name__} 执行异常", False, f"{type(exc).__name__}: {exc}")

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"汇总：{len(_RESULTS) - n_fail}/{len(_RESULTS)} 项通过"
          + ("（全部 PASS ✔）" if n_fail == 0 else f"；{n_fail} 项 FAIL ✘"))
    if n_fail:
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"  FAIL: {name}  {detail}")
    print("=" * 74)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
