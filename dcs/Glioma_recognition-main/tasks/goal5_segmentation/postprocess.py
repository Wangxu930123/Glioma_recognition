"""Goal5 的后处理：概率图 → 干净的二值掩膜。

规范 §5.1：``postprocess.py`` 属于 **[运行]** 交付文件。

为什么必须做后处理（而不只是阈值化）：
1. **孤立小斑点**：滑窗拼接边缘常出现几个体素的假阳性，直接提交会拉低
   Precision 与 HD95；
2. **多连通碎片**：真实病灶在低阈值下常碎成多块，需要保留主要成分并做
   形态学桥接，否则 Dice 会明显低于目视结果；
3. **边界层噪声**：脑外/颅骨附近的假阳性会同时污染两个掩膜。

流程：阈值化 → （可选）形态学桥接 → 按体积排序保留主要连通域 → 最小体素过滤。
为可复现，全部为确定性操作，不使用任何随机量。
"""
from __future__ import annotations

import numpy as np


def _largest_components(mask: np.ndarray, keep_n: int) -> np.ndarray:
    """保留体积最大的前 ``keep_n`` 个连通域。"""
    from scipy import ndimage

    if not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    if n <= keep_n:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1][:keep_n]
    keep = np.zeros(n + 1, dtype=bool)
    keep[order + 1] = True
    return keep[lab]


def _bridge(mask: np.ndarray, radius_mm: float, spacing: tuple[float, float, float]) -> np.ndarray:
    """形态学闭运算桥接邻近碎片（半径按 mm 给定，与体素尺寸无关）。"""
    from scipy import ndimage

    if radius_mm <= 0:
        return mask
    rad = [max(1, int(round(radius_mm / max(1e-6, s)))) for s in spacing]
    st = np.ones((2 * rad[0] + 1, 2 * rad[1] + 1, 2 * rad[2] + 1), dtype=bool)
    return ndimage.binary_closing(mask, structure=st, border_value=0)


def clean_mask(prob: np.ndarray, threshold: float, min_voxels: int,
               spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
               keep_components: int = 3, bridge_mm: float = 0.0) -> np.ndarray:
    """单通道：概率图 → 二值掩膜（确定性）。"""
    m = np.asarray(prob) > float(threshold)
    if not m.any():
        return m.astype(bool)
    if bridge_mm > 0:
        m = _bridge(m, bridge_mm, spacing)
    m = _largest_components(m, max(1, int(keep_components)))
    return m if int(m.sum()) >= int(min_voxels) else np.zeros_like(m, dtype=bool)


def clean_pair(core_prob: np.ndarray, flair_prob: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """双通道联合后处理。

    任务语义上 **core ⊆ peri**（增强核心区是周围总异常区的一部分）。
    若模型给出的 core 越出 peri，说明边界处的两通道不一致——这里用
    并集兜底，避免出现"核心区在异常区之外"这种物理上不可能的答案。
    """
    core = clean_mask(core_prob, cfg.default_thresholds[0], cfg.min_tumor_voxels,
                      keep_components=cfg.keep_components, bridge_mm=cfg.bridge_mm)
    peri = clean_mask(flair_prob, cfg.default_thresholds[1], cfg.min_tumor_voxels,
                      keep_components=cfg.keep_components, bridge_mm=cfg.bridge_mm)
    if core.any() and peri.any():
        peri = np.logical_or(peri, core)
    return core, peri
