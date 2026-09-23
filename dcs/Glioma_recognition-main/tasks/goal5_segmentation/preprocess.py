"""Goal5 的确定性预处理：把 ``Study`` 变成 1mm 公共网格上的多通道体积。

规范 §5.1：``preprocess.py`` 属于 **[运行]** 交付文件，必须是**确定性**的
（不含随机增强），且不得扫描比赛目录——它只接收已加载的领域对象 ``Study``。

设计要点：
1. **固定 1mm 公共网格**：不同序列的层厚常不一致（如 T1C 1mm、FLAIR 3mm），
   必须统一到同一网格，否则多通道无法对齐、掩膜也无法写回；
2. **逐通道独立 z-score**：MRI 强度无绝对物理意义，跨序列做全局归一化会破坏对比；
3. **缺模态用零通道占位**并以同序返回，让下游保持通道语义稳定。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.series_selector import select as select_series
from data.structures import Study
from tasks.goal5_segmentation.config import Goal5Config
from tasks.goal5_segmentation.spatial import resample_to, spacing_of, target_grid

#: 通道顺序固定，与训练时的 in_channels 一一对应（改动即破坏权重兼容）
CHANNEL_ORDER: tuple[str, ...] = ("t1c", "flair", "t2", "t1")

#: 参考网格的模态优先级（选层厚最接近 1mm 的那个作为基准）
_REF_PRIORITY: tuple[str, ...] = ("t1c", "t1", "flair", "t2")


@dataclass(frozen=True)
class PreparedVolume:
    """预处理结果：公共网格体积 + 几何 + 各通道来源信息。"""

    volume: np.ndarray                     # [C, D, H, W] float32
    affine: np.ndarray                     # 公共网格 voxel→RAS
    shape: tuple[int, int, int]
    channel_sources: dict[str, dict]       # {模态: {"series_uid":..., "spacing":...}}
    missing: tuple[str, ...]               # 缺失的通道（已用零占位）


def _zscore(vol: np.ndarray, clip: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """按前景体素做 z-score（只统计非零区域，避免空气主导统计量）。"""
    fg = vol[vol > 0]
    if fg.size < 16:
        return vol.astype(np.float32)
    lo, hi = np.percentile(fg, list(clip))
    x = np.clip(vol, lo, hi)
    m, s = float(x[vol > 0].mean()), float(x[vol > 0].std())
    return ((x - m) / (s if s > 1e-6 else 1.0)).astype(np.float32)


def build_volume(study: Study, cfg: Goal5Config) -> PreparedVolume:
    """把 ``Study`` 归一化到 1mm 公共网格，返回多通道体积。

    Raises:
        ValueError: 该 Study 没有任何可用影像（规范 §9.1：属**不可降级**输入错误）。
    """
    picked = select_series(study, CHANNEL_ORDER)
    if not picked:
        # 与 tasks/_common/volume.py 保持同一口径：报错要能自证"看到了什么序列"，
        # 否则只能看到一句"无任何可用序列"，无法判断是命名问题还是数据缺模态。
        seen = [(s.series_uid, s.modality) for s in list(study.series)[:6]]
        from data.modality_fallback import describe_sources

        raise ValueError(
            f"study {study.accession_number!r} 无任何可用序列"
            f"（共 {len(study.series)} 条；uid/描述前几条={seen}）。"
            f"若 uid 是哈希或 DICOM UID，说明序列类型没读到："
            f"确认数据根下有 SeriesType.xlsx 或同名 .json sidecar"
            f"（已自动尝试转模态识别：{describe_sources()}）"
        )

    # 1) 选参考网格：优先 1mm 附近的模态，且必须在实际存在的序列中选择
    ref_key = next((k for k in _REF_PRIORITY if k in picked), next(iter(picked)))
    ref = picked[ref_key]
    grid_shape, grid_affine = target_grid(ref.image.shape, ref.affine,
                                          tuple(cfg.common_spacing))

    # 2) 逐通道重采样 + 归一化
    chans: list[np.ndarray] = []
    sources: dict[str, dict] = {}
    missing: list[str] = []
    for name in CHANNEL_ORDER:
        s = picked.get(name)
        if s is None:
            chans.append(np.zeros(grid_shape, dtype=np.float32))
            missing.append(name)
            continue
        arr = np.asarray(s.image, dtype=np.float32)
        if arr.shape != tuple(grid_shape) or not np.allclose(s.affine, grid_affine):
            arr = resample_to(arr, s.affine, grid_shape, grid_affine, order=1)
        chans.append(_zscore(arr))
        sources[name] = {
            "series_uid": s.series_uid,
            "spacing": spacing_of(s.affine),
            "shape": tuple(int(x) for x in s.image.shape),
        }

    vol = np.stack(chans, axis=0).astype(np.float32)
    return PreparedVolume(volume=vol, affine=np.asarray(grid_affine, dtype=np.float64),
                          shape=tuple(int(x) for x in grid_shape),
                          channel_sources=sources, missing=tuple(missing))
