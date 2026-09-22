#!/usr/bin/env python
"""故障注入鲁棒性压测：验证**任何离谱输入**都仍产出合规答案。

规范硬性要求：“每个 AccessionNumber 都必须有 prediction.json”。
本脚本构造 ~20 类异常输入（文件损坏 / 模态缺失 / 几何越界 / 数值畸形 / 掩码异常），
逐例断言三件事：

  1. ``GliomaPipeline.run_batch`` 本身不抛异常（整批不因单例而中断）；
  2. 数据集里**每一个** AccessionNumber 都有 prediction.json（完整性，最关键）；
  3. 整体通过 ``validate_answer``（几何/数值/字段全部合规）。

用法::

    python scripts/22_fault_injection.py [--ckpt PATH] [--keep]

退出码 0 表示全部通过。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SZ = 48                                     # 基准体积边长（小体积 → 单次滑窗，跑得快）


def _affine(spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    a = np.diag([float(spacing[0]), float(spacing[1]), float(spacing[2]), 1.0])
    return a


def _nii(path: str, arr: np.ndarray, affine: np.ndarray | None = None) -> None:
    import nibabel as nib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = nib.Nifti1Image(np.asarray(arr), _affine() if affine is None else affine)
    nib.save(img, path)


def _tumor(sz: int = SZ, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """含一个亮块的合成影像（保证正常病例能被推理）。"""
    v = np.zeros((sz, sz, sz), np.float32)
    lo, hi = sz // 4, sz // 2
    v[lo:hi, lo:hi, lo:hi] = 200.0
    v += np.random.RandomState(0).randn(sz, sz, sz).astype(np.float32) * 3.0
    return v


def _case_dir(root: str, acc: str, uid: str = "S1") -> tuple[str, str]:
    d = os.path.join(root, acc, uid)
    os.makedirs(d, exist_ok=True)
    return d, os.path.join(root, acc)


# --------------------------------------------------------------------------- #
# 故障场景定义：每个函数在 root/acc/ 下造出"畸形输入"
# --------------------------------------------------------------------------- #
def s_normal(root: str, acc: str) -> None:
    """对照组：完好病例。"""
    d, ad = _case_dir(root, acc)
    _nii(f"{d}/t1c.nii.gz", _tumor())
    _nii(f"{d}/flair.nii.gz", _tumor())
    _nii(f"{ad}/mask_core.nii.gz", (_tumor() > 50).astype(np.uint8))
    _nii(f"{ad}/mask_flair.nii.gz", (_tumor() > 50).astype(np.uint8))


def s_corrupt(root: str, acc: str) -> None:
    """影像文件内容损坏（随机字节，nibabel 无法解析）。"""
    d, ad = _case_dir(root, acc)
    with open(f"{d}/t1c.nii.gz", "wb") as f:
        f.write(np.random.RandomState(1).bytes(4096))
    _nii(f"{d}/flair.nii.gz", _tumor())
    _nii(f"{ad}/mask_core.nii.gz", (_tumor() > 50).astype(np.uint8))


def s_truncated(root: str, acc: str) -> None:
    """合法 gzip 头 + 被截断的正文。"""
    d, ad = _case_dir(root, acc)
    tmp = f"{d}/_full.nii.gz"
    _nii(tmp, _tumor())
    data = open(tmp, "rb").read()
    os.remove(tmp)
    with open(f"{d}/t1c.nii.gz", "wb") as f:
        f.write(data[: int(len(data) * 0.3)])
    _nii(f"{d}/flair.nii.gz", _tumor())
    _nii(f"{ad}/mask_core.nii.gz", (_tumor() > 50).astype(np.uint8))


def s_empty_file(root: str, acc: str) -> None:
    """0 字节文件。"""
    d, ad = _case_dir(root, acc)
    open(f"{d}/t1c.nii.gz", "wb").close()
    _nii(f"{d}/flair.nii.gz", _tumor())
    _nii(f"{ad}/mask_core.nii.gz", (_tumor() > 50).astype(np.uint8))


def s_all_zero(root: str, acc: str) -> None:
    """影像全 0（无脑组织）。"""
    d, ad = _case_dir(root, acc)
    _nii(f"{d}/t1c.nii.gz", np.zeros((SZ, SZ, SZ), np.float32))
    _nii(f"{d}/flair.nii.gz", np.zeros((SZ, SZ, SZ), np.float32))
    _nii(f"{ad}/mask_core.nii.gz", np.zeros((SZ, SZ, SZ), np.uint8))


def s_all_nan(root: str, acc: str) -> None:
    """影像全 NaN（会让 z-score / 归一化产生 NaN 并可能污染 GPU 推理）。"""
    d, ad = _case_dir(root, acc)
    nan = np.full((SZ, SZ, SZ), np.nan, np.float32)
    _nii(f"{d}/t1c.nii.gz", nan)
    _nii(f"{d}/flair.nii.gz", nan.copy())
    _nii(f"{ad}/mask_core.nii.gz", np.zeros((SZ, SZ, SZ), np.uint8))


def s_with_inf(root: str, acc: str) -> None:
    """影像含 ±inf。"""
    d, ad = _case_dir(root, acc)
    v = _tumor()
    v[0, 0, 0] = np.inf
    v[1, 1, 1] = -np.inf
    _nii(f"{d}/t1c.nii.gz", v)
    _nii(f"{d}/flair.nii.gz", v.copy())
    _nii(f"{ad}/mask_core.nii.gz", (v > 50).astype(np.uint8))


def s_only_flair(root: str, acc: str) -> None:
    """只有 FLAIR，缺 T1C（core 掩码无法回写）。"""
    d, ad = _case_dir(root, acc)
    _nii(f"{d}/flair.nii.gz", _tumor())
    _nii(f"{ad}/mask_flair.nii.gz", (_tumor() > 50).astype(np.uint8))


def s_only_t1c(root: str, acc: str) -> None:
    """只有 T1C，缺 FLAIR。"""
    d, ad = _case_dir(root, acc)
    _nii(f"{d}/t1c.nii.gz", _tumor())
    _nii(f"{ad}/mask_core.nii.gz", (_tumor() > 50).astype(np.uint8))


def s_no_image(root: str, acc: str) -> None:
    """目录里只有掩码，没有任何影像。"""
    _d, ad = _case_dir(root, acc)
    _nii(f"{ad}/mask_core.nii.gz", (_tumor() > 50).astype(np.uint8))


def s_empty_dir(root: str, acc: str) -> None:
    """完全空的病例目录。"""
    os.makedirs(os.path.join(root, acc), exist_ok=True)


def s_tiny_spacing(root: str, acc: str) -> None:
    """spacing 0.01mm（体素极小 → 重采样后体积会爆炸）。"""
    d, ad = _case_dir(root, acc)
    _nii(f"{d}/t1c.nii.gz", _tumor(24), _affine((0.01, 0.01, 0.01)))
    _nii(f"{d}/flair.nii.gz", _tumor(24), _affine((0.01, 0.01, 0.01)))
    _nii(f"{ad}/mask_core.nii.gz", np.zeros((24, 24, 24), np.uint8))


def s_huge_spacing(root: str, acc: str) -> None:
    """spacing 20mm（体素极大 → 重采样后只剩几个体素）。"""
    d, ad = _case_dir(root, acc)
    _nii(f"{d}/t1c.nii.gz", _tumor(16), _affine((20.0, 20.0, 20.0)))
    _nii(f"{d}/flair.nii.gz", _tumor(16), _affine((20.0, 20.0, 20.0)))
    _nii(f"{ad}/mask_core.nii.gz", np.zeros((16, 16, 16), np.uint8))


def s_single_slice(root: str, acc: str) -> None:
    """单切片体积（D=1）。"""
    d, ad = _case_dir(root, acc)
    _nii(f"{d}/t1c.nii.gz", np.ones((1, 64, 64), np.float32) * 100)
    _nii(f"{d}/flair.nii.gz", np.ones((1, 64, 64), np.float32) * 100)
    _nii(f"{ad}/mask_core.nii.gz", np.zeros((1, 64, 64), np.uint8))


def s_4d_volume(root: str, acc: str) -> None:
    """4D 影像（带时间轴）。"""
    d, ad = _case_dir(root, acc)
    v = np.stack([_tumor(32)] * 3, -1)
    _nii(f"{d}/t1c.nii.gz", v)
    _nii(f"{d}/flair.nii.gz", v.copy())
    _nii(f"{ad}/mask_core.nii.gz", np.zeros((32, 32, 32), np.uint8))


def s_singular_affine(root: str, acc: str) -> None:
    """affine 为全 0 奇异矩阵（不可逆）。"""
    d, ad = _case_dir(root, acc)
    bad = np.zeros((4, 4), float)
    _nii(f"{d}/t1c.nii.gz", _tumor(32), bad)
    _nii(f"{d}/flair.nii.gz", _tumor(32), bad)
    _nii(f"{ad}/mask_core.nii.gz", np.zeros((32, 32, 32), np.uint8))


def s_nan_affine(root: str, acc: str) -> None:
    """affine 含 NaN。"""
    d, ad = _case_dir(root, acc)
    bad = np.diag([np.nan, 1.0, 1.0, 1.0])
    _nii(f"{d}/t1c.nii.gz", _tumor(32), bad)
    _nii(f"{d}/flair.nii.gz", _tumor(32), bad)
    _nii(f"{ad}/mask_core.nii.gz", np.zeros((32, 32, 32), np.uint8))


def s_tiny_volume(root: str, acc: str) -> None:
    """2×2×2 的极小体积。"""
    d, ad = _case_dir(root, acc)
    _nii(f"{d}/t1c.nii.gz", np.ones((2, 2, 2), np.float32))
    _nii(f"{d}/flair.nii.gz", np.ones((2, 2, 2), np.float32))
    _nii(f"{ad}/mask_core.nii.gz", np.zeros((2, 2, 2), np.uint8))


def s_mask_shape_mismatch(root: str, acc: str) -> None:
    """掩码维度与影像不符（32³ vs 48³）。"""
    d, ad = _case_dir(root, acc)
    _nii(f"{d}/t1c.nii.gz", _tumor(SZ))
    _nii(f"{d}/flair.nii.gz", _tumor(SZ))
    _nii(f"{ad}/mask_core.nii.gz", np.ones((32, 32, 32), np.uint8))


def s_brats_labels(root: str, acc: str) -> None:
    """BraTS 风格多类标签（0/1/2/4）当作掩码，且 core/flair 写在同一目录下。"""
    d, ad = _case_dir(root, acc)
    _nii(f"{d}/t1c.nii.gz", _tumor())
    _nii(f"{d}/flair.nii.gz", _tumor())
    lab = np.zeros((SZ, SZ, SZ), np.uint8)
    lab[10:20, 10:20, 10:20] = 1
    lab[20:30, 20:30, 20:30] = 2
    lab[30:36, 30:36, 30:36] = 4
    _nii(f"{ad}/seg_brats.nii.gz", lab)


def s_weird_accession(root: str, acc: str) -> None:
    """AccessionNumber 含空格与特殊字符（目录名即 Acc）。"""
    d, _ad = _case_dir(root, acc)
    _nii(f"{d}/t1c.nii.gz", _tumor())
    _nii(f"{d}/flair.nii.gz", _tumor())


def s_huge_intensity(root: str, acc: str) -> None:
    """强度值 1e10（远超 int16 范围，模拟未归一化的原始 DICOM 值）。"""
    d, ad = _case_dir(root, acc)
    v = _tumor().astype(np.float32) * 1e8
    _nii(f"{d}/t1c.nii.gz", v)
    _nii(f"{d}/flair.nii.gz", v.copy())
    _nii(f"{ad}/mask_core.nii.gz", (v > 1e9).astype(np.uint8))


SCENARIOS: list[tuple[str, object, str | None]] = [
    ("正常对照", s_normal, None),
    ("影像损坏(随机字节)", s_corrupt, None),
    ("影像被截断", s_truncated, None),
    ("0字节文件", s_empty_file, None),
    ("影像全0", s_all_zero, None),
    ("影像全NaN", s_all_nan, None),
    ("影像含inf", s_with_inf, None),
    ("缺T1C(仅FLAIR)", s_only_flair, None),
    ("缺FLAIR(仅T1C)", s_only_t1c, None),
    ("无影像仅有掩码", s_no_image, None),
    ("空病例目录", s_empty_dir, None),
    ("spacing=0.01mm", s_tiny_spacing, None),
    ("spacing=20mm", s_huge_spacing, None),
    ("单切片D=1", s_single_slice, None),
    ("4D影像", s_4d_volume, None),
    ("affine全0", s_singular_affine, None),
    ("affine含NaN", s_nan_affine, None),
    ("2x2x2极小体积", s_tiny_volume, None),
    ("掩码维度不符", s_mask_shape_mismatch, None),
    ("BraTS多类标签", s_brats_labels, None),
    ("Acc含特殊字符", s_weird_accession, "P &特殊-001_CT"),
    ("强度1e10", s_huge_intensity, None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint 路径（默认自动挑一个）")
    ap.add_argument("--keep", action="store_true", help="保留临时目录")
    args = ap.parse_args()

    from src.data.probe import scan_real
    from src.inference.pipeline import GliomaPipeline
    from src.utils.config import load_config

    ckpt = args.ckpt
    if not ckpt:
        for tag in ("g4_fold0", "g4_fold1", "g4_fold2", "smoke_fold0"):
            p = os.path.join(ROOT, "checkpoints", tag, "best.pth")
            if os.path.exists(p):
                ckpt = p
                break
    if not ckpt or not os.path.exists(ckpt):
        print(f"[fault] 找不到 checkpoint（--ckpt 指定）：{ckpt}")
        return 2

    work = tempfile.mkdtemp(prefix="glioma_fault_")
    ds = os.path.join(work, "dataset")
    out = os.path.join(work, "answer", "fault_0001")
    os.makedirs(ds, exist_ok=True)

    # 每个场景一个 AccessionNumber；名称保持规范友好（只含字母数字下划线）
    accs: dict[str, str] = {}
    print("=" * 78)
    print(f"故障注入压测：{len(SCENARIOS)} 个场景    权重={os.path.relpath(ckpt, ROOT)}")
    print("=" * 78)
    for i, (name, fn, acc_override) in enumerate(SCENARIOS):
        acc = acc_override or f"FAULT{i:03d}"
        accs[acc] = name
        try:
            fn(ds, acc)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [build] {acc} {name}: 构造异常 {exc}")

    print(f"[fault] 数据集 {len(accs)} 例 → {ds}")
    print("[fault] 运行推理流水线（整批不得因单例中断）…")

    import torch
    pipe = GliomaPipeline([ckpt], device="cuda" if torch.cuda.is_available() else "cpu")
    try:
        res = pipe.run_batch(ds, out, cases=None)
        crashed = None
    except Exception as exc:                                       # noqa: BLE001
        import traceback
        traceback.print_exc()
        res, crashed = None, exc

    # ---------------- 断言 ---------------- #
    print("\n" + "=" * 78)
    print("逐例结果")
    print("=" * 78)
    print(f"{'Accession':<12}{'场景':<22}{'探针识别':<10}{'有答案':<8}{'自检'}")
    print("-" * 78)

    try:
        seen_accs = {c.get("accession") for c in scan_real(ds, None, {}, []) or []}
    except Exception:                                              # noqa: BLE001
        seen_accs = set()
    n_missing, covered = [], 0
    for acc, name in accs.items():
        pjson = os.path.join(out, acc, "prediction.json")
        has = os.path.exists(pjson)
        if has:
            covered += 1
        else:
            n_missing.append((acc, name))
        seen = acc in seen_accs
        state = "-"
        if has:
            try:
                d = json.load(open(pjson, encoding="utf-8"))
                fields = load_config("labels.yaml")["fields"]
                miss = [f["key"] for f in fields if f["key"] not in d.get("Prediction", {})]
                state = "OK" if not miss else f"缺字段{len(miss)}"
            except Exception as exc:                                # noqa: BLE001
                state = f"JSON异常:{type(exc).__name__}"
        print(f"{acc:<12}{name:<22}{str(seen):<10}{str(has):<8}{state}")

    print("-" * 78)
    ok = True
    if crashed is not None:
        print(f"✗ run_batch 整批抛异常：{type(crashed).__name__}: {crashed}")
        ok = False
    else:
        print(f"✓ run_batch 未抛异常（整批完成）")

    if n_missing:
        print(f"✗ 有 {len(n_missing)} 例**没有 prediction.json**（规范要求每例必须有）：")
        for acc, name in n_missing:
            print(f"    - {acc}  {name}")
        ok = False
    else:
        print(f"✓ 全部 {len(accs)} 例都有 prediction.json")

    if res is not None:
        rep = res.get("validate") or {}
        print(f"{'✓' if rep.get('ok') else '✗'} validate_answer ok={rep.get('ok')} "
              f"errors={len(rep.get('errors') or [])}")
        for e in (rep.get("errors") or [])[:8]:
            print(f"    ! {e}")
        if not rep.get("ok"):
            ok = False
        print(f"  完成 {res.get('n_done')}/{res.get('n_cases')} 例；"
              f"走兜底 {len(res.get('failed') or [])} 例")

    print("\n" + "=" * 78)
    print(("全部通过 ✔  任何异常输入均产出合规答案" if ok else "存在问题 ✘ 见上方"))
    print("=" * 78)
    if args.keep:
        print(f"[fault] 保留工作目录：{work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
