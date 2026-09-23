"""数据流水线（赛道4）：多序列 MRI → 1mm 公共网格 → 双通道分割目标 + 结构化标签。

设计要点（相对早期版本的关键修正）
1. **真正的公共网格**：所有序列重采样到 **1mm 各向同性**（非"以首个序列网格为参考"）。
   若原始层厚过大（> ``grid.max_spacing_factor`` × 目标），该轴保留原始 spacing，
   避免 z 轴过度插值产生大量相关假层。用 ``nibabel.processing`` 做重采样，
   彻底避免 SimpleITK 与 nibabel 轴序不一致导致的错位。
2. **掩码角色按序列模态绑定**（重要）：数据集中 T1C 目录下的"肿瘤瘤体"是任务A(core)，
   FLAIR/T2 目录下的"瘤体/水肿/全肿瘤"是任务B(peri)；只看文件名会把 FLAIR 上的
   "瘤体"误判成 core 并写错空间。同一角色多个掩码取**并集**。
3. **缺失序列降级**：T1C 缺失用 T1/T2 顶替，FLAIR 缺失用 T2（日志记录）。
4. **强增广**：翻转 / 90°旋转 / 随机缩放 / 弹性形变 / 强度扰动 / 偏置场 / 噪声，
   显著提升小样本泛化（BraTS 类任务上通常 +2~4 Dice）。
5. **特殊影像与重复影像**：`SpecialImageDataset` 训练假人体/拼接头（目标一二），
   `DuplicatePairDataset` 训练嵌入（目标一二的重复影像）。
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.config import load_config, resolve
from .labels import describe_modality_sources, guess_modality, mask_role_for

# --------------------------------------------------------------------------- #
# 几何：公共网格 / 重采样（nibabel.processing，affine 直连，无轴序陷阱）
# --------------------------------------------------------------------------- #
def load_nii(path: str) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib
    img = nib.load(path)
    return np.asanyarray(img.dataobj), np.asarray(img.affine, dtype=float)


def resample_to_grid(arr: np.ndarray, affine: np.ndarray, out_shape: tuple,
                     out_affine: np.ndarray, order: int = 1) -> np.ndarray:
    """把 (arr, affine) 重采样到 (out_shape, out_affine) 网格；几何由 affine 直连。"""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    src = nib.Nifti1Image(np.asarray(arr, dtype=np.float32), np.asarray(affine, float))
    ref = nib.Nifti1Image(np.zeros(tuple(out_shape), dtype=np.float32), np.asarray(out_affine, float))
    out = resample_from_to(src, ref, order=order, mode="constant", cval=0.0)
    return np.asanyarray(out.dataobj)


def target_grid(ref_shape: tuple, ref_affine: np.ndarray, cfg: dict
                ) -> tuple[tuple, np.ndarray]:
    """由参考序列推出公共网格（1mm 各向同性，方向与参考序列一致）。

    - 保持参考序列的方向余弦（D）与原点，只替换 spacing → 与源影像轴序一致；
    - 若某轴原始 spacing 远大于目标（如层厚 5mm），该轴保留原始 spacing。
    """
    g = cfg.get("geometry", {})
    want = np.asarray(g.get("common_spacing", [1.0, 1.0, 1.0]), float)
    max_factor = float(g.get("max_spacing_factor", 1.5))
    sp = np.array([float(np.linalg.norm(np.asarray(ref_affine)[:3, i])) for i in range(3)])
    sp[sp <= 0] = 1.0
    tgt = np.where(sp <= max_factor * want, want, sp)          # 层厚过大的轴保持原样
    if np.allclose(tgt, sp, rtol=1e-3, atol=1e-4):             # 已在目标网格 → 原样使用
        return tuple(ref_shape), np.asarray(ref_affine, float)
    D = np.asarray(ref_affine, float)[:3, :3] / sp[None, :]
    shape = tuple(int(np.ceil(ref_shape[i] * sp[i] / tgt[i])) for i in range(3))
    aff = np.eye(4)
    aff[:3, :3] = D * tgt[None, :]
    aff[:3, 3] = np.asarray(ref_affine, float)[:3, 3]
    return shape, aff


def zscore_volume(arr: np.ndarray, clip=(0.5, 99.5), foreground_only: bool = True) -> np.ndarray:
    """按体积 z-score；``foreground_only`` 时只用非零体素估计均值/方差（MRI 标准做法）。"""
    a = arr.astype(np.float32)
    sel = a[a > 0] if (foreground_only and (a > 0).any()) else a.ravel()
    if sel.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(sel, clip)
    sel = np.clip(sel, lo, hi)
    m, s = float(sel.mean()), float(sel.std())
    b = np.clip(a, lo, hi)
    return ((b - m) / (s + 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 病例 → 多通道体数据 + 目标
# --------------------------------------------------------------------------- #
#: 统计模态判别模型的状态（``data/modality_model.json``）。
#: **懒加载且只尝试一次**：常规路径（序列名带模态 / 有官方标注表）永远不会碰它，
#: 因此对常规训练与推理零开销；评测集的 UID 目录名才会走到这里。
_MODEL_STATE: dict[str, Any] = {"loaded": False, "model": None}

#: 模型输出的官方模态 → 内部通道名（与 ``CHANNEL_ORDER`` 一致）
_MODEL_TO_CHANNEL = {"T1CE": "t1c", "T1CE ": "t1c", "T2": "t2", "FLAIR": "flair"}


def _modality_model():
    if not _MODEL_STATE["loaded"]:
        _MODEL_STATE["loaded"] = True
        try:
            from .modality_model import load_default_model
            _MODEL_STATE["model"] = load_default_model()
        except Exception:                                         # noqa: BLE001
            _MODEL_STATE["model"] = None
    return _MODEL_STATE["model"]


def classify_unknown(case: dict, cfg: dict, log: list | None = None) -> dict[str, dict]:
    """把"认不出模态"的序列交给统计模型判别 → ``{通道名: meta}``。

    **为什么必须有这条路**：官方训练集给了 ``3_serieslabel.xlsx``，评测集不给；
    评测集的序列目录名是 DICOM UID，任何关键词都命中不了。此时若不判模态：
    训练侧表现为"无任何可用序列"直接崩，推理侧更隐蔽 ——
    ``inference/pipeline.py`` 要先知道"哪个序列是 T1C"才能把掩码写回它的空间，
    认不出就写不回去，提交上去的掩码空间是错的（评测端直接判错）。

    依赖 ``case["unknown_series"]``（``data.probe`` 产出，已剔除能靠名字认出的序列）。
    """
    unknown = case.get("unknown_series") or []
    if not unknown:
        return {}
    model = _modality_model()
    if model is None:
        if log is not None:
            log.append(f"{case.get('accession')}: 有 {len(unknown)} 路序列模态未知，"
                       f"但未找到 data/modality_model.json → 无法判别"
                       f"（用 scripts/31_train_modality_model.py 训练）")
        return {}

    wanted = [ch["name"] for ch in cfg["channels"]]
    probs: list[tuple[dict, dict[str, float]]] = []
    for meta in unknown:
        try:
            _label, prob = model.predict_file(str(meta["path"]))
        except Exception as exc:                                  # noqa: BLE001
            if log is not None:
                log.append(f"{case.get('accession')}: 模态判别失败 {meta.get('file')}（{exc}）")
            continue
        probs.append((meta, prob))
    if not probs:
        return {}

    # **全局贪心的一对一指派**（而不是逐路"先到先得"）：
    # 一路序列只能是一个模态，但"先到先得"在 argmax 撞车时会白丢一路 ——
    # 真值 (T1CE,T2,FLAIR) 被判成 (FLAIR,T2,FLAIR) 时，第三路会被整个丢弃，
    # 于是 t1c 通道空着，而它其实还能靠**次优**标签救回来。
    # 把所有 (序列, 通道) 候选按概率降序逐个占用，即可拿到"尽量对齐"的指派。
    cand: list[tuple[float, int, str]] = []
    for si, (_meta, prob) in enumerate(probs):
        for label, p in prob.items():
            name = _MODEL_TO_CHANNEL.get(str(label).strip().upper())
            if name in wanted:
                cand.append((float(p), si, name))
    cand.sort(reverse=True)

    out: dict[str, dict] = {}
    used_series: set[int] = set()
    for p, si, name in cand:
        # 置信度不足宁可不填：把 FLAIR 当成 T1C 会把水肿送进"增强核心区"通道，
        # 比留一个空通道更有害（空通道至少是明确的"缺失"）。候选已降序 → 可直接停。
        if p < 0.5:
            break
        if name in out or si in used_series:
            continue
        meta, _prob = probs[si]
        out[name] = {**meta, "modality": name,
                     "from_model": name, "model_confidence": round(p, 3)}
        used_series.add(si)
        if log is not None:
            log.append(f"{case.get('accession')}: 模态判别 → {name}"
                       f"（{os.path.basename(str(meta['path']))}，置信 {p:.2f}）")
    if log is not None and not out:
        log.append(f"{case.get('accession')}: {len(probs)} 路序列模态判别置信度均 <0.5 → "
                   f"全部弃用（宁可缺通道，也不把模态填错）")
    return out


def pick_series(case: dict, cfg: dict, log: list | None = None) -> dict[str, dict]:
    """按 channels 定义（含 fallback）决定每个通道实际使用的序列。

    返回 ``{通道名: {"modality","path","series_uid"}}``；推理侧据此决定
    "掩码应写回哪个序列的空间"（core→T1C、peri→FLAIR/T2）。
    """
    out: dict[str, dict] = {}
    for ch in cfg["channels"]:
        for cand in [ch["name"]] + list(ch.get("fallback") or []):
            if cand in case.get("images", {}):
                meta = dict(case["images"][cand])
                meta["modality"] = cand
                out[ch["name"]] = meta
                break
    # 名字全都对不上 → 用体素统计模型判（评测集 UID 目录名走这里）
    if len(out) < len(cfg["channels"]):
        for name, meta in classify_unknown(case, cfg, log).items():
            out.setdefault(name, meta)
    return out


def _mask_variants(role_entry: Any) -> list[dict]:
    """兼容 manifest 的两种掩码写法：{"path":...} 或 {"paths":[...],"metas":[...]}。"""
    if role_entry is None:
        return []
    if isinstance(role_entry, dict) and role_entry.get("paths"):
        metas = role_entry.get("metas") or [{} for _ in role_entry["paths"]]
        return [{"path": p, **m} for p, m in zip(role_entry["paths"], metas)]
    if isinstance(role_entry, dict) and role_entry.get("path"):
        return [role_entry]
    if isinstance(role_entry, list):
        return [m for m in role_entry if isinstance(m, dict)]
    return []


def build_case_volume(case: dict, cfg: dict, log: list | None = None
                      ) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 ``(vol[C,D,H,W] float32 z-score, 公共网格 affine, 该网格上的掩码 dict)``。

    掩码角色语义（严格对齐规范）：
    - ``core``：任务A —— T1 增强核心区；
    - ``peri``：任务B —— FLAIR/T2 总异常区（全肿瘤 = 瘤体 ∪ 水肿）；
    - ``abn`` ：非肿瘤性病变（脑梗死等）的异常信号（检测负样本的弱标签）。
    """
    ch_defs = cfg["channels"]
    picked = pick_series(case, cfg, log)
    if not picked:
        # 报出"清单里到底有哪些序列键"：键全为 other/空，说明模态没认出来
        # （官方数据靠 labels/3_serieslabel.xlsx），而不是这张检查真的没影像。
        # 顺带自检两种模态来源的可用性，并给出可直接粘贴的两条命令 —— 见
        # docs/DATASET_ROOT_TROUBLESHOOT.md「病例数正常、却报无任何可用序列」。
        imgs = case.get("images") or {}
        raise RuntimeError(
            f"病例 {case['accession']} 无任何可用序列"
            f"（清单里的序列键={sorted(imgs)[:8]}；"
            f"未知序列 {len(case.get('unknown_series') or [])} 路）。"
            f"模态来源自检：{describe_modality_sources(case.get('dir'))}。"
            f"按顺序试：① export GLIOMA_LABELS_DIR=<含 3_serieslabel.xlsx 的目录>"
            f"（或 ln -s 到 <工程>/labels），然后重跑 bash scripts/01_probe.sh 与 "
            f"bash scripts/02_build_dataset.sh；"
            f"② 没有标注表时训练体素判别模型："
            f"python3 scripts/31_train_modality_model.py --root <数据根>"
            f"（产出 data/modality_model.json）；"
            f"③ 详见 docs/DATASET_ROOT_TROUBLESHOOT.md「病例数正常、却报无任何可用序列」"
        )

    # 参考序列优先级：t1c → flair → t2 → t1 → 其它（决定公共网格方向与原点）
    ref_key = next((k for k in ("t1c", "flair", "t2", "t1") if k in picked),
                   next(iter(picked)))
    ref_arr, ref_aff = load_nii(picked[ref_key]["path"])
    shape, aff = target_grid(tuple(ref_arr.shape), ref_aff, cfg)

    planes, used = [], {}
    for ch in ch_defs:
        meta = picked.get(ch["name"])
        if meta is None:
            planes.append(np.zeros(shape, dtype=np.float32))
            if log is not None:
                log.append(f"{case['accession']}: 缺 {ch['name']}（无 fallback）→ 置零通道")
            continue
        if meta["modality"] != ch["name"] and log is not None:
            log.append(f"{case['accession']}: {ch['name']} 缺失 → 用 {meta['modality']} 顶替")
        arr, a = load_nii(meta["path"])
        if tuple(arr.shape) != tuple(shape) or not np.allclose(a, aff, atol=1e-3):
            arr = resample_to_grid(arr, a, shape, aff, order=int(cfg["geometry"]["resample_order_img"]))
        planes.append(zscore_volume(arr, tuple(cfg["intensity"]["clip_percentile"]),
                                    bool(cfg["intensity"].get("foreground_only", True))))
        used[ch["name"]] = meta["modality"]
    vol = np.stack(planes, axis=0).astype(np.float32)

    # 掩码 → 公共网格（最近邻；同角色多掩码取并集）
    masks: dict[str, np.ndarray] = {}
    for role, entry in (case.get("masks") or {}).items():
        acc = None
        for meta in _mask_variants(entry):
            try:
                arr, a = load_nii(meta["path"])
            except Exception:                                     # noqa: BLE001
                continue
            m = (arr > 0).astype(np.float32)
            if tuple(m.shape) != tuple(shape) or not np.allclose(a, aff, atol=1e-3):
                m = resample_to_grid(m, a, shape, aff,
                                     order=int(cfg["geometry"]["resample_order_mask"]))
            m = (m > 0.5).astype(np.uint8)
            acc = m if acc is None else (acc | m)
        if acc is not None:
            masks[role] = acc
    return vol, aff, masks


def make_targets(masks: dict, shape: tuple) -> np.ndarray:
    """规范两目标：``core``(任务A) 与 ``peri``(任务B)。"""
    core = masks.get("core")
    peri = masks.get("peri")
    out = np.zeros((2,) + tuple(shape), dtype=np.float32)
    if core is not None:
        out[0] = core
    if peri is not None:
        out[1] = peri
        if core is not None:
            out[1] = np.maximum(out[1], out[0])                   # peri ⊇ core 硬约束
    elif core is not None:
        out[1] = core                                             # 只有 core 时弱标签兜底
    return out


# --------------------------------------------------------------------------- #
# 增广
# --------------------------------------------------------------------------- #
def _rng_np(rng: random.Random) -> np.random.Generator:
    """由 ``random.Random`` 派生 numpy 生成器（保证可复现且线程安全）。"""
    return np.random.default_rng(rng.getrandbits(32))


def _elastic(v: np.ndarray, t: np.ndarray, rng: random.Random, alpha: float, sigma: float):
    from scipy.ndimage import gaussian_filter, map_coordinates
    shape = v.shape[1:]
    nrng = _rng_np(rng)
    dz = [gaussian_filter((nrng.random(shape) * 2 - 1), sigma, mode="nearest") * alpha
          for _ in range(3)]
    zz, yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                             indexing="ij")
    coords = [np.clip(zz + dz[0], 0, shape[0] - 1),
              np.clip(yy + dz[1], 0, shape[1] - 1),
              np.clip(xx + dz[2], 0, shape[2] - 1)]
    for c in range(v.shape[0]):
        v[c] = map_coordinates(v[c], coords, order=1, mode="nearest")
    for c in range(t.shape[0]):
        t[c] = (map_coordinates(t[c], coords, order=0, mode="nearest") > 0.5).astype(np.float32)
    return v, t


def _bias_field(v: np.ndarray, rng: random.Random, strength: float):
    from scipy.ndimage import gaussian_filter
    sh = v.shape[1:]
    low = gaussian_filter(_rng_np(rng).random(sh).astype(np.float32), max(2.0, min(sh) / 4.0))
    low = (low - low.min()) / (float(np.ptp(low)) + 1e-6)
    field = 1.0 + strength * (low - 0.5) * 2.0                    # [1-s, 1+s]
    return v * field[None], None


def augment(v: np.ndarray, t: np.ndarray, rng: random.Random, cfg: dict
            ) -> tuple[np.ndarray, np.ndarray]:
    """训练增广（几何 + 强度）；掩码同步几何变换并保持二值。"""
    a = cfg.get("augment", {}) or {}
    if not a.get("enabled", True):
        return v, t

    # 1) 翻转 / 90° 旋转（解剖对称性安全）
    for ax in (1, 2, 3):
        if rng.random() < float(a.get("flip_prob", 0.5)):
            v = np.flip(v, ax).copy()
            t = np.flip(t, ax).copy()
    k = rng.randrange(4)
    if k:
        axes = (1, 2) if float(a.get("rot_xy_prob", 1.0)) > rng.random() else (2, 3)
        v = np.rot90(v, k, axes).copy()
        t = np.rot90(t, k, axes).copy()

    # 2) 随机缩放（各轴独立，温和范围）
    if rng.random() < float(a.get("scale_prob", 0.3)):
        from scipy.ndimage import zoom
        f = [1.0 + rng.uniform(-1, 1) * float(a.get("scale_range", 0.15)) for _ in range(3)]
        v = np.stack([zoom(v[c], f, order=1, mode="nearest") for c in range(v.shape[0])])
        t = np.stack([zoom(t[c], f, order=0, mode="nearest") for c in range(t.shape[0])])
        t = (t > 0.5).astype(np.float32)

    # 3) 弹性形变（低分辨率位移场，模拟解剖变异）
    if rng.random() < float(a.get("elastic_prob", 0.25)):
        v, t = _elastic(v, t, rng, float(a.get("elastic_alpha", 2.0)),
                        float(a.get("elastic_sigma", 8.0)))

    # 4) 强度扰动（只作用于图像通道）
    v = augment_intensity(v, rng, cfg)
    return v.astype(np.float32), t.astype(np.float32)


def augment_intensity(v: np.ndarray, rng: random.Random, cfg: dict) -> np.ndarray:
    """只做强度扰动（gamma / 尺度 / 位移 / 偏置场 / 噪声）——全局视图复用。"""
    a = cfg.get("augment", {}) or {}
    if not a.get("enabled", True):
        return v
    if rng.random() < float(a.get("intensity_prob", 0.8)):
        g = 1.0 + rng.uniform(-1, 1) * float(a.get("gamma_range", 0.25))
        s = 1.0 + rng.uniform(-1, 1) * float(a.get("scale_range_i", 0.15))
        b = rng.uniform(-1, 1) * float(a.get("shift_range", 0.15))
        v = np.sign(v) * np.power(np.abs(v) + 1e-6, g) * s + b
    if rng.random() < float(a.get("bias_prob", 0.2)):
        v, _ = _bias_field(v, rng, float(a.get("bias_strength", 0.25)))
    if rng.random() < float(a.get("noise_prob", 0.3)):
        v = v + rng.uniform(0.0, 1.0) * float(a.get("noise_std", 0.08)) \
            * _rng_np(rng).standard_normal(v.shape).astype(np.float32)
    return v.astype(np.float32)


def crop_patch(vol: np.ndarray, tgt: np.ndarray, patch: tuple, rng: random.Random,
               train: bool, pos_ratio: float, jitter: int = 0):
    """按 patch 大小取块；训练时按 ``pos_ratio`` 保证含病灶的概率。"""
    c, d, h, w = vol.shape
    pd, ph, pw = patch
    pos = tgt[1] > 0 if tgt[1].sum() > 0 else (tgt[0] > 0)
    if train and pos.any() and rng.random() < pos_ratio:
        idx = np.argwhere(pos)
        ctr = np.asarray(idx[rng.randrange(len(idx))], dtype=int)
    elif pos.any() and not train:
        idx = np.argwhere(pos)
        ctr = np.asarray(idx[len(idx) // 2], dtype=int)
    else:
        ctr = np.array([d // 2, h // 2, w // 2])
    ctr = np.asarray(ctr, dtype=int)[:3]
    starts, pads = [], []
    for ci, (sz, ps) in enumerate(zip((d, h, w), (pd, ph, pw))):
        j = rng.randint(-jitter, jitter) if (train and jitter) else 0
        s = int(ctr[ci] - ps // 2 + j)
        s = max(0, min(s, max(0, sz - ps)))
        starts.append(s)
        pads.append(max(0, ps - sz))
    sl = tuple(slice(s, s + ps) for s in starts)
    v = vol[(slice(None),) + sl]
    t = tgt[(slice(None),) + sl]
    if any(pads):
        pad = ((0, 0),) + tuple((0, p) for p in pads)
        v = np.pad(v, pad)
        t = np.pad(t, pad)
    return v, t


# --------------------------------------------------------------------------- #
# 全局视图（分类 / 特殊影像 / 重复影像嵌入）——**训练与推理尺度必须一致**
# --------------------------------------------------------------------------- #
def global_view(vol: np.ndarray, center: np.ndarray | None, size_mm: float, out: int,
                spacing: float = 1.0) -> np.ndarray:
    """取固定**物理尺寸**立方体（默认以病灶质心为中心）并缩放到 out³。

    早期版本推理用"整脑降采样"、训练用"局部 patch"，两者物理尺度不同
    （如推理 1.9mm/体素 vs 训练 1.0mm/体素），分类头性能会显著退化。
    这里统一为：物理立方体 ``size_mm`` → ``out³``，推理侧同样处理。
    """
    from scipy.ndimage import zoom
    c, d, h, w = vol.shape
    half = size_mm / max(1e-6, spacing) / 2.0
    ctr = np.asarray(center, float) if center is not None else np.array([d / 2, h / 2, w / 2])
    sl, pad_lo = [], []
    for i, sz in enumerate((d, h, w)):
        s0 = int(round(ctr[i] - half))
        lo = max(0, -s0)
        s0 = max(0, s0)
        s1 = min(sz, int(round(ctr[i] + half)))
        need = int(round(2 * half)) - (s1 - s0)
        hi = max(0, need - lo)
        sl.append((s0, s1))
        pad_lo.append((lo, hi))
    sub = vol[(slice(None),) + tuple(slice(a, b) for a, b in sl)]
    pad = ((0, 0),) + tuple(pad_lo)
    if any(p != (0, 0) for p in pad[1:]):
        sub = np.pad(sub, pad)
    f = [1.0] + [out / max(1, s) for s in sub.shape[1:]]
    return zoom(sub, f, order=1).astype(np.float32)


def lesion_center(tgt: np.ndarray) -> np.ndarray | None:
    """病灶质心（优先 peri，其次 core）；无病灶返回 None。"""
    m = tgt[1] > 0
    if not m.any():
        m = tgt[0] > 0
    if not m.any():
        return None
    return np.asarray(np.argwhere(m).mean(0), float)


def brain_center(vol: np.ndarray) -> np.ndarray | None:
    """非零体素包围盒中心（推理时无掩码可用）。"""
    nz = np.argwhere(np.abs(vol).sum(0) > 1e-3)
    if len(nz) == 0:
        return None
    lo, hi = nz.min(0), nz.max(0)
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
def resolve_cache_dir(cfg: dict | None = None) -> str | None:
    """预处理缓存目录（``CACHE_DIR``/``paths.preprocess_cache``）；不存在则返回 None。"""
    p = os.environ.get("CACHE_DIR") or os.environ.get("PREPROCESS_CACHE")
    if not p:
        try:
            from ..utils.config import load_paths
            p = load_paths().get("preprocess_cache")
        except Exception:                                          # noqa: BLE001
            p = None
    if not p:
        return None
    p = resolve(p)
    return p if os.path.isdir(p) else None


def build_volume_from_arrays(available: dict[str, tuple[np.ndarray, np.ndarray]],
                             cfg: dict, log: list | None = None
                             ) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """**跨工程桥接入口**：从内存数组构建公共网格体数据。

    团队串联工程（``Glioma_recognition``）的 ``Series`` 已经把 NIfTI 读进内存
    （``image`` + voxel-to-RAS ``affine``），因此这里不再读盘，直接接收
    ``{模态: (image, affine)}``，按 ``cfg["channels"]``（含 fallback）组装多通道体积。

    返回 ``(vol[C,D,H,W] float32 z-score, 公共网格 affine, {通道: 实际使用的模态})``。
    实际使用的模态用于把掩码逆变换回**正确的那条源序列**的空间。
    """
    picked: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for ch in cfg["channels"]:
        for cand in [ch["name"]] + list(ch.get("fallback") or []):
            item = available.get(cand)
            if item is not None and item[0] is not None:
                picked[ch["name"]] = (cand, np.asarray(item[0]), np.asarray(item[1], float))
                break
    if not picked:
        raise RuntimeError("没有任何可用序列（t1c/flair/t2/t1 全部缺失）")

    ref_key = next((k for k in ("t1c", "flair", "t2", "t1") if k in picked),
                   next(iter(picked)))
    ref_arr, ref_aff = picked[ref_key][1], picked[ref_key][2]
    shape, aff = target_grid(tuple(ref_arr.shape), ref_aff, cfg)

    planes, used = [], {}
    for ch in cfg["channels"]:
        item = picked.get(ch["name"])
        if item is None:
            planes.append(np.zeros(shape, dtype=np.float32))
            if log is not None:
                log.append(f"缺 {ch['name']}（无 fallback）→ 置零通道")
            continue
        mod, arr, a = item
        if mod != ch["name"] and log is not None:
            log.append(f"{ch['name']} 缺失 → 用 {mod} 顶替")
        if tuple(arr.shape) != tuple(shape) or not np.allclose(a, aff, atol=1e-3):
            arr = resample_to_grid(arr, a, shape, aff,
                                   order=int(cfg["geometry"]["resample_order_img"]))
        planes.append(zscore_volume(arr, tuple(cfg["intensity"]["clip_percentile"]),
                                    bool(cfg["intensity"].get("foreground_only", True))))
        used[ch["name"]] = mod
    return np.stack(planes, axis=0).astype(np.float32), aff, used


def load_case_patch(case: dict, cache_dir: str, patch: tuple, rng: random.Random,
                    train: bool, pos_ratio: float, jitter: int = 0
                    ) -> tuple[np.ndarray, np.ndarray] | None:
    """**按需切片读取**缓存（mmap）：只读 patch 覆盖的体素，避免每例全量载入 71MB。

    缓存在机械盘时这是决定训练吞吐的关键（实测每步 I/O 从 142MB 降到 ~21MB）。
    返回 ``(image[C,pd,ph,pw] float32, target[2,pd,ph,pw] float32)``；
    缓存缺失或元信息不全时返回 ``None``（调用方回退到全量路径）。
    """
    d = os.path.join(cache_dir, str(case["accession"]))
    vp, cpth, ppth, mp = (os.path.join(d, n) for n in
                          ("vol.npy", "core.npy", "peri.npy", "meta.json"))
    if not (os.path.isfile(vp) and os.path.isfile(mp)):
        return None
    try:
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
        core = np.load(cpth, mmap_mode="r")
        peri = np.load(ppth, mmap_mode="r")
        vol = np.load(vp, mmap_mode="r")
    except Exception:                                             # noqa: BLE001
        return None
    sh = tuple(int(x) for x in meta.get("shape", core.shape)[1:])
    if len(sh) != 3 or any(s <= 0 for s in sh):
        return None
    pd, ph, pw = patch

    # ---- 选 crop 中心 ----
    ctr = meta.get("lesion_center")
    if train and ctr is not None and rng.random() < pos_ratio:
        c = np.asarray(ctr, float) + np.array([rng.randint(-12, 12) for _ in range(3)])
    elif (not train) and ctr is not None:
        c = np.asarray(ctr, float)
    elif train:
        c = np.array([rng.uniform(0.25, 0.75) * s for s in sh])
    else:
        c = np.array([s / 2 for s in sh])
    c = np.asarray(c, float)[:3]

    starts, pads = [], []
    for i, (sz, ps) in enumerate(zip(sh, patch)):
        j = rng.randint(-jitter, jitter) if (train and jitter) else 0
        s0 = int(round(c[i] - ps / 2)) + j
        # 参考实现：先尝试居中，再夹紧到 [0, sz-ps]；若 sz<ps 则 pad
        if sz <= ps:
            s0 = 0
            pads.append(ps - sz)
        else:
            s0 = max(0, min(s0, sz - ps))
            pads.append(0)
        starts.append(s0)
    sl = tuple(slice(s, s + ps) for s, ps in zip(starts, patch))
    try:
        v = np.asarray(vol[(slice(None),) + sl], np.float32)
        c_ = np.asarray(core[sl], np.uint8)
        p_ = np.asarray(peri[sl], np.uint8)
    except Exception:                                             # noqa: BLE001
        return None
    if any(pads):
        pad = ((0, 0),) + tuple((0, q) for q in pads)
        v = np.pad(v, pad)
        pad2 = tuple((0, q) for q in pads)
        c_ = np.pad(c_, pad2)
        p_ = np.pad(p_, pad2)
    t = np.stack([(c_ > 0).astype(np.float32), (p_ > 0).astype(np.float32)])
    return v, t


def load_case_cached(case: dict, cfg: dict, cache_dir: str | None = None
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    """读病例（公共网格体积 / affine / 掩码）；优先磁盘缓存，未命中则现场构建。

    缓存由 ``scripts/13_build_cache.py`` 生成（float16 + uint8，体积减半、读取极快），
    是训练吞吐的关键（实测每例 2.2s → 0.2s 量级）。
    """
    if cache_dir:
        d = os.path.join(cache_dir, str(case["accession"]))
        vp = os.path.join(d, "vol.npy")
        if os.path.isfile(vp):
            try:
                vol = np.load(vp).astype(np.float32)
                masks = {"core": np.load(os.path.join(d, "core.npy")),
                         "peri": np.load(os.path.join(d, "peri.npy"))}
                with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                    meta = json.load(f)
                return vol, np.asarray(meta["affine"], float), masks
            except Exception:                                      # noqa: BLE001
                pass
    return build_case_volume(case, cfg)


def load_whole_cached(case: dict, cfg: dict, cache_dir: str | None = None) -> np.ndarray:
    """整脑视图（固定物理尺寸 → out³）；优先缓存。"""
    if cache_dir:
        wp = os.path.join(cache_dir, str(case["accession"]), "whole.npy")
        if os.path.isfile(wp):
            try:
                return np.load(wp).astype(np.float32)
            except Exception:                                      # noqa: BLE001
                pass
    gv = cfg.get("global_view") or {}
    vol, _aff, masks = build_case_volume(case, cfg)
    return global_view(vol, lesion_center(make_targets(masks, vol.shape[1:])),
                       float(gv.get("size_mm", 192)), int(gv.get("out", 96)))


class GliomaDataset(Dataset):
    """逐 patch 采样；返回多通道图像、2 通道分割目标、整脑视图、结构化标签与 label_mask。"""

    def __init__(self, cases: list[dict], train: bool = True, patch: tuple = (96, 96, 96),
                 pos_ratio: float = 0.7, pre_cfg: dict | None = None, label_fields: list | None = None,
                 seed: int = 42, aug_cfg: dict | None = None, cache_size: int = 8,
                 cache_dir: str | None = None):
        self.cases = cases
        self.train = train
        self.patch = tuple(patch)
        self.pos_ratio = pos_ratio
        self.cfg = pre_cfg or load_config("preprocess.yaml")
        self.aug_cfg = aug_cfg or load_config("train.yaml")
        self.fields = label_fields or []
        self.rng = random.Random(seed)
        self._cache: dict[str, Any] = {}
        self._cache_size = cache_size
        self.cache_dir = cache_dir if cache_dir is not None else resolve_cache_dir()
        self._wcache: dict[str, Any] = {}

    def __len__(self) -> int:
        return max(1, len(self.cases))

    def _load(self, case: dict):
        key = case["accession"]
        if key in self._cache:
            return self._cache[key]
        vol, aff, masks = load_case_cached(case, self.cfg, self.cache_dir)
        tgt = make_targets(masks, vol.shape[1:])
        self._cache[key] = (vol, tgt, aff)
        if len(self._cache) > self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        return vol, tgt, aff

    def _whole(self, case: dict, vol: np.ndarray | None = None,
               tgt: np.ndarray | None = None) -> np.ndarray:
        key = case["accession"]
        if key in self._wcache:
            return self._wcache[key]
        gv = self.aug_cfg.get("global_view") or {}
        w = load_whole_cached(case, {**self.cfg, "global_view": gv}, self.cache_dir) \
            if self.cache_dir else global_view(vol, lesion_center(tgt),
                                               float(gv.get("size_mm", 192)),
                                               int(gv.get("out", 96)))
        self._wcache[key] = w
        if len(self._wcache) > max(16, self._cache_size * 4):
            self._wcache.pop(next(iter(self._wcache)))
        return w

    def __getitem__(self, i: int) -> dict:
        case = self.cases[i % len(self.cases)]
        # 快路径：mmap 按需切片（只读 patch 覆盖的体素，避免全量载入）
        got = load_case_patch(case, self.cache_dir, self.patch, self.rng, self.train,
                              self.pos_ratio) if self.cache_dir else None
        if got is not None:
            v, t = got
            whole = self._whole(case, None, None)
        else:
            vol, tgt, _aff = self._load(case)
            v, t = crop_patch(vol, tgt, self.patch, self.rng, self.train, self.pos_ratio)
            whole = self._whole(case, vol, tgt)
        if self.train:
            v, t = augment(v, t, self.rng, self.aug_cfg)
            v, t = _fit_patch(v, t, self.patch, self.rng)         # 尺寸对齐
            whole = augment_intensity(whole, self.rng, self.aug_cfg)

        labels = np.zeros(len(self.fields), dtype=np.float32)
        lmask = np.zeros(len(self.fields), dtype=np.float32)
        for fi, f in enumerate(self.fields):
            val = case.get("labels", {}).get(f["key"])
            if val is None:
                continue
            if f["type"] == "binary":
                labels[fi] = 1.0 if int(val) else 0.0
            else:
                classes = [str(c) for c in f["classes"]]
                text = str(val)
                if text in classes:
                    labels[fi] = classes.index(text)
                else:
                    # 官方 `5_characteristics.xlsx` 的 ``Location`` 是**多标签**
                    # （``|`` 分隔，如 ``LeftTemporal|LeftFrontal``），而比赛枚举是单值。
                    # 旧实现直接 ``continue``：这些样本的该字段被**静默 mask 掉**，
                    # Location 头实际拿不到任何监督（且不报错）。
                    # 这里取**第一个能匹配上的标签**——保留监督且确定。
                    tokens = [t.strip() for t in re.split(r"[|,;/、]", text) if t.strip()]
                    hit = next((t for t in tokens if t in classes), None)
                    if hit is None:
                        lower = {c.lower(): c for c in classes}
                        hit = next((lower[t.lower()] for t in tokens
                                    if t.lower() in lower), None)
                    if hit is None:
                        continue
                    labels[fi] = classes.index(hit)
            lmask[fi] = 1.0
        return {"image": torch.from_numpy(np.ascontiguousarray(v)),
                "target": torch.from_numpy(np.ascontiguousarray(t)),
                "whole": torch.from_numpy(np.ascontiguousarray(whole)),
                "labels": torch.from_numpy(labels), "label_mask": torch.from_numpy(lmask),
                "accession": case["accession"]}


def _fit_patch(v: np.ndarray, t: np.ndarray, patch: tuple, rng: random.Random):
    """把（增广后尺寸可能变化的）patch 对齐回固定 patch 尺寸。"""
    out_v = np.zeros((v.shape[0],) + tuple(patch), dtype=np.float32)
    out_t = np.zeros((t.shape[0],) + tuple(patch), dtype=np.float32)
    sl_src, sl_dst = [], []
    for i in range(3):
        cur, want = v.shape[i + 1], patch[i]
        if cur >= want:
            s = rng.randint(0, cur - want) if rng.random() < 0.5 else (cur - want) // 2
            sl_src.append(slice(s, s + want)); sl_dst.append(slice(0, want))
        else:
            sl_src.append(slice(0, cur)); sl_dst.append(slice(0, cur))
    src = (slice(None),) + tuple(sl_src)
    dst = (slice(None),) + tuple(sl_dst)
    out_v[dst] = v[src]
    out_t[dst] = t[src]
    return out_v, out_t


class _WholeViewMixin:
    """把病例转成"固定物理尺寸整脑视图"（与推理同尺度），带缓存与强度增广。"""

    def _init_whole(self, pre_cfg: dict | None, aug_cfg: dict | None, cache_size: int,
                    seed: int, cache_dir: str | None = None):
        self.cfg = pre_cfg or load_config("preprocess.yaml")
        self.aug_cfg = aug_cfg or load_config("train.yaml")
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self.rng = random.Random(seed)
        self.cache_dir = cache_dir if cache_dir is not None else resolve_cache_dir()

    def _whole(self, case: dict) -> np.ndarray:
        acc = case["accession"]
        if acc in self._cache:
            return self._cache[acc]
        gv = self.aug_cfg.get("global_view") or {}
        out = int(gv.get("out", 96))
        try:
            v = load_whole_cached(case, {**self.cfg, "global_view": gv}, self.cache_dir)
        except Exception:                                          # noqa: BLE001
            v = np.zeros((len(self.cfg["channels"]), out, out, out), np.float32)
        self._cache[acc] = v
        if len(self._cache) > self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        return v


class SpecialImageDataset(_WholeViewMixin, Dataset):
    """目标一/二：假人体(fake) 与 拼接影像(Composition) 二分类。

    - 正样本：``annotation/fake``、``annotation/Composition`` 内的病例；
    - 负样本：其余正常影像病例（规范明确：三目录之外的病例同时是两者的阴性）；
    - **label_mask**：fake 只监督通道0、Composition 只监督通道1，正常病例同时监督两者
      （避免把一个任务的正样本当作另一个任务的负样本）。
    """

    def __init__(self, cases: list[dict], pos_fake: set[str], pos_composition: set[str],
                 pre_cfg: dict | None = None, aug_cfg: dict | None = None,
                 seed: int = 42, n_per_epoch: int = 1024, cache_size: int = 64,
                 pos_ratio: float = 0.5):
        self.cases = [c for c in cases if c.get("images")]
        self.by_acc = {c["accession"]: c for c in self.cases}
        self.pos_fake = [a for a in pos_fake if a in self.by_acc]
        self.pos_comp = [a for a in pos_composition if a in self.by_acc]
        self.neg = [c["accession"] for c in self.cases
                    if c["accession"] not in set(pos_fake) | set(pos_composition)]
        self.pos_ratio = pos_ratio
        self.n_per_epoch = n_per_epoch
        self._init_whole(pre_cfg, aug_cfg, cache_size, seed)

    def __len__(self) -> int:
        return max(self.n_per_epoch, 1)

    def __getitem__(self, i: int) -> dict:
        task = i % 2                                                # 0=fake, 1=composition
        pos_list = self.pos_fake if task == 0 else self.pos_comp
        want_pos = bool(pos_list) and (self.rng.random() < self.pos_ratio)
        acc = self.rng.choice(pos_list if want_pos else (self.neg or pos_list))
        label = 1.0 if acc in pos_list else 0.0
        v = self._whole(self.by_acc[acc])
        v = augment_intensity(v, self.rng, self.aug_cfg)
        if self.rng.random() < 0.5:                                 # 轻几何（不改变物理尺度）
            ax = self.rng.randrange(3) + 1
            v = np.flip(v, ax).copy()
        tgt = np.zeros(2, dtype=np.float32)
        lmask = np.zeros(2, dtype=np.float32)
        tgt[task] = label
        lmask[task] = 1.0
        return {"image": torch.from_numpy(np.ascontiguousarray(v)),
                "target": torch.from_numpy(tgt), "label_mask": torch.from_numpy(lmask),
                "accession": acc}


class DuplicatePairDataset(_WholeViewMixin, Dataset):
    """重复影像配对训练：金标准对 → 正样本；随机配对 → 负样本。

    规范指标为 AUC-PR / Recall@10%FPR / Precision@15%Recall，
    因此负样本比例设置偏高（``n_neg_per_pos``），让模型学会区分"像但不同"的检查。
    """

    def __init__(self, cases: list[dict], gold_pairs: list[list[str]], n_neg_per_pos: int = 3,
                 seed: int = 42, pre_cfg: dict | None = None, aug_cfg: dict | None = None,
                 cache_size: int = 128):
        self.by_acc = {c["accession"]: c for c in cases if c.get("images")}
        self.gold = [[a, b] for a, b in gold_pairs if a in self.by_acc and b in self.by_acc]
        self.accs = sorted(self.by_acc)
        self.n_neg = max(1, int(n_neg_per_pos))
        self._init_whole(pre_cfg, aug_cfg, cache_size, seed)

    def __len__(self) -> int:
        return max(1, len(self.gold) * (1 + self.n_neg))

    def _aug(self, v: np.ndarray) -> np.ndarray:
        v = augment_intensity(v, self.rng, self.aug_cfg)
        if self.rng.random() < 0.5:
            v = np.flip(v, self.rng.randrange(3) + 1).copy()
        return v

    def __getitem__(self, i: int) -> dict:
        if not self.gold:
            a = b = self.rng.choice(self.accs)
            z = torch.zeros_like(torch.from_numpy(self._whole(self.by_acc[a])))
            return {"a": z, "b": z.clone(), "label": torch.tensor(1.0),
                    "acc_a": a, "acc_b": b}
        gi = (i // (1 + self.n_neg)) % len(self.gold)
        a, b = self.gold[gi]
        label = 1.0
        if i % (1 + self.n_neg) != 0:                              # 负样本（换掉第二例）
            b = self.rng.choice(self.accs)
            label = 0.0 if b != a else 1.0
        return {"a": torch.from_numpy(np.ascontiguousarray(self._aug(self._whole(self.by_acc[a])))),
                "b": torch.from_numpy(np.ascontiguousarray(self._aug(self._whole(self.by_acc[b])))),
                "label": torch.tensor(label), "acc_a": a, "acc_b": b}


# --------------------------------------------------------------------------- #
# 折划分（按"有无肿瘤标注"分层，保证每折阳性比例一致）
# --------------------------------------------------------------------------- #
def build_folds(manifest_path: str, n_folds: int = 5, val_ratio: float = 0.2,
                seed: int = 42, force: bool = False) -> dict:
    """分层 K 折划分（**每例恰好属于一个折的 val**）；写入 folds.json。

    修正说明（重要）
    ----------------
    早期实现用"轮转取模"拼接每折的 val：

        n_val_pos = round(len(pos) * val_ratio)
        val = [pos[(k * n_val_pos + j) % len(pos)] ...]

    当 ``n_folds * n_val_pos != len(pos)`` 时必然出错（``round`` 的四舍五入
    几乎总会打破这个等式）：

    * 偏多 → 取模绕回 → **同一病例出现在多个折的 val**（被重复评估）；
    * 偏少 → 有病例**从未进入任何折的 val**（永远拿不到 OOF 预测，
      且它在所有折都是训练样本）。

    实测 624 例 / 5 折下出现了 3 例重复（FAKE_000/FAKE_001/COMP_002）
    与 2 例完全缺失（BraTS2021_00085/00318）。

    现改为标准做法：对 pos / neg 分别"轮流发牌"（``items[i::n_folds]``），
    各折 val 天然**互斥**且**并集为全集**，同时每折阳性比例保持均衡
    （每折 val ≈ 1/n_folds，与 ``val_ratio=0.2, n_folds=5`` 一致）。

    ⚠️ 修改划分会让**已训练权重与其训练集不再对应**（旧折的 train 与新折的
    val 会大范围重叠 → 评估泄漏）。因此重新训练前不要覆盖 folds.json。
    """
    with open(resolve(manifest_path), encoding="utf-8") as f:
        man = json.load(f)
    cases = man["cases"]
    out = resolve(os.path.join(os.path.dirname(manifest_path), "folds.json"))
    all_accs = {c["accession"] for c in cases}

    # ★ 复用已存在的划分（除非 force=True）。
    # trainer 每次启动都会调用本函数；若无条件重写，则"重启任意一个折"就会
    # 静默换掉整套划分 → 正在跑的折用的是旧划分、重启的折用新划分，
    # 交叉验证与 OOF 评估全部失效（且不会报任何错）。
    #
    # 但**只有"覆盖当前数据"的旧划分才允许复用**。早期实现只看折数：
    # 换到官方数据后，磁盘上那份来自本地模拟集的 5 折文件折数正好相符 →
    # 直接原样返回，于是
    #   · 02_build_dataset.sh 打印的是**新清单**的病例数（看着像重建过了），
    #   · 折划分却仍是旧数据集的病例号（零重叠）；
    #   · 训练端一路提示"与当前数据不匹配"并退回按比例划分，
    #     六个人各训一折的交叉验证从此不成立。
    # 现在多一道覆盖性校验，不匹配就重建并把原因说清楚。
    if os.path.exists(out) and not force:
        try:
            with open(out, encoding="utf-8") as f:
                existing = json.load(f)
            if len(existing) == n_folds:
                covered: set[str] = set()
                for entry in existing.values():
                    covered |= set((entry or {}).get("val") or [])
                if covered == all_accs:
                    return existing
                print(f"[folds] 已有划分与当前数据**不匹配** → 重建："
                      f"旧划分覆盖 {len(covered)} 例、当前 {len(all_accs)} 例、"
                      f"交集仅 {len(covered & all_accs)} 例。"
                      f"常见原因：folds.json 来自**另一个数据根**"
                      f"（如换到官方数据后未重建），或数据清单被换过。", flush=True)
            else:
                print(f"[folds] 已有划分折数={len(existing)} 与请求 n_folds={n_folds} 不符 → 重建")
        except Exception as exc:                                  # noqa: BLE001
            print(f"[folds] 读取已有划分失败（{exc}）→ 重建")

    pos = [c["accession"] for c in cases if c.get("masks") or c.get("labels")]
    neg = [c["accession"] for c in cases if c["accession"] not in set(pos)]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def _partition(items: list[str]) -> list[list[str]]:
        """轮流发牌 → n_folds 份，互斥且并集为全集（大小最多差 1）。"""
        return [items[i::n_folds] for i in range(n_folds)]

    pos_parts, neg_parts = _partition(pos), _partition(neg)
    folds: dict[str, dict] = {}
    for k in range(n_folds):
        val = sorted(set(pos_parts[k]) | set(neg_parts[k]))
        folds[str(k)] = {"val": val, "train": sorted(all_accs - set(val))}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=1)
    return folds
