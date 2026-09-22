"""共享的多通道体积预处理：``Study`` → 1mm 公共网格。

Goal1/2/3/4/5 都需要把检查的多个序列对齐到同一网格后再喂给共享骨干，
因此该逻辑属于**公共模块**（规范 §5.1 禁止每个 Goal 各写一套 NIfTI 读取）。

设计要点：
1. **固定 1mm 公共网格**：不同序列层厚常不一致（T1C 1mm / FLAIR 3mm），
   不统一网格则多通道无法对齐、掩膜也无法写回；
2. **逐通道独立 z-score**：MRI 强度无绝对物理意义；
3. **缺模态零占位**并保持通道顺序固定，让下游语义稳定。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.selector import pick_series as select_series
from shared.spatial import resample_to, spacing_of, target_grid

#: 通道顺序固定，与训练时 in_channels 一一对应（改动即破坏权重兼容）
CHANNEL_ORDER: tuple[str, ...] = ("t1c", "flair", "t2", "t1")

#: 参考网格的模态优先级
_REF_PRIORITY: tuple[str, ...] = ("t1c", "t1", "flair", "t2")


@dataclass(frozen=True)
class PreparedVolume:
    """预处理结果：公共网格体积 + 几何 + 各通道来源信息。"""

    volume: np.ndarray                     # [C, D, H, W] float32
    affine: np.ndarray                     # 公共网格 voxel→RAS
    shape: tuple[int, int, int]
    channel_sources: dict[str, dict]       # {模态: {"series_uid":..., "spacing":...}}
    missing: tuple[str, ...]               # 缺失的通道（已用零占位）


def zscore(vol: np.ndarray, clip: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """按前景体素做 z-score（只统计非零区域，避免空气主导统计量）。"""
    fg = vol[vol > 0]
    if fg.size < 16:
        return vol.astype(np.float32)
    lo, hi = np.percentile(fg, list(clip))
    x = np.clip(vol, lo, hi)
    m, s = float(x[vol > 0].mean()), float(x[vol > 0].std())
    return ((x - m) / (s if s > 1e-6 else 1.0)).astype(np.float32)


def build_volume(study, common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 channels: tuple[str, ...] = CHANNEL_ORDER) -> PreparedVolume:
    """把 ``Study`` 归一化到公共网格，返回多通道体积。

    Raises:
        ValueError: 该 Study 没有任何可用影像（规范 §9.1：不可降级输入错误）。
    """
    picked = select_series(study, channels)
    if not picked:
        raise ValueError(f"study {study.accession_number!r} 无任何可用序列")

    ref_key = next((k for k in _REF_PRIORITY if k in picked), next(iter(picked)))
    ref = picked[ref_key]
    grid_shape, grid_affine = target_grid(ref.image.shape, ref.affine, tuple(common_spacing))

    chans: list[np.ndarray] = []
    sources: dict[str, dict] = {}
    missing: list[str] = []
    for name in channels:
        s = picked.get(name)
        if s is None:
            chans.append(np.zeros(grid_shape, dtype=np.float32))
            missing.append(name)
            continue
        arr = np.asarray(s.image, dtype=np.float32)
        if arr.shape != tuple(grid_shape) or not np.allclose(s.affine, grid_affine):
            arr = resample_to(arr, s.affine, grid_shape, grid_affine, order=1)
        chans.append(zscore(arr))
        sources[name] = {
            "series_uid": s.series_uid,
            "spacing": spacing_of(s.affine),
            "shape": tuple(int(x) for x in s.image.shape),
        }

    vol = np.stack(chans, axis=0).astype(np.float32)
    return PreparedVolume(volume=vol, affine=np.asarray(grid_affine, dtype=np.float64),
                          shape=tuple(int(x) for x in grid_shape),
                          channel_sources=sources, missing=tuple(missing))


def global_view(vol: np.ndarray, size_mm: float = 192.0, out: int = 96,
                spacing: float = 1.0, center: np.ndarray | None = None) -> np.ndarray:
    """取固定**物理尺寸**立方体并缩放到 ``out³``。

    分类/特殊影像/嵌入这三个"全局头"在训练时看到的是固定物理尺度的整脑视图，
    推理侧必须用同样尺度，否则训练与推理的输入分布不一致、全局头性能退化。
    """
    from scipy.ndimage import zoom

    c, d, h, w = vol.shape
    half = float(size_mm) / max(1e-6, float(spacing)) / 2.0
    if center is None:
        nz = np.argwhere(np.abs(vol).sum(0) > 1e-3)
        ctr = ((nz.min(0) + nz.max(0)) / 2.0) if len(nz) else np.array([d / 2, h / 2, w / 2])
    else:
        ctr = np.asarray(center, float)

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
