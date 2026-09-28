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

from data.series_selector import guess_modality as _guess_modality
from data.series_selector import select as select_series
from data.structures import Series, Study
from tasks._common.spatial import resample_to, spacing_of, target_grid

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


def _no_usable_series_message(study) -> str:
    """「没有任何可用影像」的报错原文：带上"看到了什么" + 两类成因的区分。

    只报一句"没有任何可用影像"时，既不知道序列叫什么、也无从判断是**命名/路径问题**
    还是**该检查本来就没有目标模态**，而这段堆栈还常常埋在 DataLoader worker 里
    （看不出是路径问题）。因此把两类信息都拼进报错：

    - 每条序列的 ``uid`` / ``描述``，并标注描述的性质：
      ``未解析（描述退化成目录名/UID）`` = 数据信息表与 sidecar 都没给值；
      ``描述里没有模态关键词`` = 拿到了值（表/sidecar）但不是目标模态（如 ``其他``）；
    - 提醒另一半口径：**只有 `其他` 序列 / 只有 DWI 的检查不再走这条路**
      （全放开口径下会"全零通道 + 借几何"照走，见 glioma_track4 README §7.2.1），
      所以这里失败基本等于"这个检查连一路影像都没有"。

    ⚠️ 两个工程各有一份 ``volume.py``（不合并），审计要求**实现同源**，
    因此本函数体内**不能**出现各自工程特有的模块路径 —— 项目相关的辅助函数
    一律走模块级别的别名 import（见 :data:`_guess_modality`）。
    """
    seen = []
    for s in list(study.series)[:6]:
        uid = str(getattr(s, "series_uid", "?") or "?")
        desc = str(getattr(s, "modality", "") or "")
        if not desc or desc == uid:
            note = "未解析（描述退化成目录名/UID）"
        elif _guess_modality(desc) is None:
            note = "描述里没有模态关键词"
        else:
            note = ""
        seen.append((uid, desc, note) if note else (uid, desc))
    return (
        f"study {study.accession_number!r} 没有任何可用影像"
        f"（共 {len(study.series)} 条序列；uid/描述前几条={seen}）。"
        f"注意：**只有 `其他` 序列 / 只有 DWI 的检查不算这一类**，"
        f"那种情况会全零通道照走。"
        f"若 uid 是哈希或 DICOM UID，说明序列类型没读到：确认数据根下有 "
        f"SeriesType.xlsx 或同名 .json sidecar，"
        f"或 export GLIOMA_LABELS_DIR=<它所在目录>；"
        f"详见 glioma_track4/docs/DATASET_ROOT_TROUBLESHOOT.md"
        f"「病例数正常、却报无任何可用序列」"
    )


def _any_series_ref(study: Study) -> "Series | None":
    """该 Study 里**任意一路有影像**的序列（几何参考用），没有则 ``None``。

    用途：4 个通道一路都填不上时（整例序列被数据信息表标成 `其他`，或只有
    DWI/ADC/SWI），公共网格仍需要一个参考几何 —— 而**几何与模态无关**，
    任意一路序列的 affine/shape 都能把网格建出来，掩膜也才有地方重采样。
    """
    for s in getattr(study, "series", None) or ():
        if getattr(s, "image", None) is not None and getattr(s, "affine", None) is not None:
            return s
    return None


def build_volume(study: Study, common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 channels: tuple[str, ...] = CHANNEL_ORDER) -> PreparedVolume:
    """把 ``Study`` 归一化到公共网格，返回多通道体积。

    4 个通道**一个都填不上**时不再抛错（**全放开口径**，与算法工程
    ``glioma_track4`` 一致，见那里 README §7.2.1）：改为"全零通道 + 借任意一路序列
    的几何"。这类检查的成因是序列被数据信息表标成 `其他`（或只有 DWI/ADC/SWI）——
    它们不属于这 4 个通道，但**影像与掩膜都在**、掩膜也仍在正确的任务空间里
    （掩膜角色只依赖模态），丢整例就是白丢数据。代价是这一例回传近噪声梯度，
    所以 ``missing`` 会把"四个通道全缺"暴露出来，不静默。

    Raises:
        ValueError: 该 Study **连一路影像都没有**（规范 §9.1：不可降级输入错误）。
            注意这与"没有目标模态"是两回事：后者现在会全零通道照走。
    """
    picked = select_series(study, channels)
    if not picked:
        ref = _any_series_ref(study)
        if ref is None:
            raise ValueError(_no_usable_series_message(study))
    else:
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
