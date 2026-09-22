"""Goal5 的分割配置。

规范 §5.2：本文件只声明**相对路径**，绝对根目录由 ``core.config.Settings.ckpt_root``
提供，模型代码中不得出现任何硬编码的比赛路径。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Goal5Config:
    #: 相对 checkpoint 根目录的权重路径（规范 §5.2 约定）
    core_ckpt_rel: str = "goal5_segmentation/core.pt"
    flair_ckpt_rel: str = "goal5_segmentation/flair.pt"

    #: 网络结构（必须与训练时一致，否则权重加载失败）
    arch: str = "mednext"
    in_channels: int = 4
    base: int = 32
    depth: int = 4
    blocks_per_stage: int = 2
    k: int = 3
    expand: int = 2
    aniso_z: bool = False
    max_ch: int = 320

    #: 形态学后处理
    min_tumor_voxels: int = 30
    keep_components: int = 3
    bridge_mm: float = 10.0

    #: 公共网格（1mm）与滑窗
    common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    patch: tuple[int, int, int] = (96, 96, 96)
    overlap: float = 0.5
    tta_flips: tuple[str, ...] = ("x", "y")
    tta_batch: int = 2
    global_size: int = 96
    global_size_mm: float = 192.0

    #: 二值化阈值；训练完成后由 14_calibrate_thresholds.py 写回权重，
    #: 加载时以权重里记录的阈值为准，这里的默认值只作兜底。
    default_thresholds: tuple[float, float] = (0.5, 0.5)

    #: 通道语义（与训练时的 seg 通道顺序一致，改动即破坏权重兼容）
    core_channel: int = 0
    flair_channel: int = 1
