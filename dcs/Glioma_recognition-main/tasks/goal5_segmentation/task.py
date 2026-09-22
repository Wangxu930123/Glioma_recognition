"""Goal5 的 Task 适配器：比赛 Pipeline 的**唯一插件入口**（规范 §5.1）。

职责边界（严格对齐规范）：
- 只接收领域对象 ``PipelineContext``，返回强类型 ``Goal5Result``；
- **不**直接读取比赛根路径、**不**写 ``answer/``、**不**发回调；
- **不**导入任何训练专用模块（dataset/augmentations/losses/train/evaluate）；
- 完成推理与**逆变换**，把掩膜恢复到各自源序列空间——Writer 只负责落盘。

输出语义：``core_mask`` 对应 T1C 序列空间，``flair_mask`` 对应 FLAIR/T2 序列空间；
两者的 shape/affine 必须与来源序列完全一致，否则该例分割会被判 0 分。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.config import Settings
from data.series_selector import infer_uid_from_path, select as select_series
from tasks.base import StudyTask
from tasks.goal5_segmentation.config import Goal5Config
from tasks.goal5_segmentation.inference import LoadedGoal5, infer_segmentation, load_model
from tasks.goal5_segmentation.postprocess import clean_pair
from tasks.goal5_segmentation.preprocess import PreparedVolume, build_volume
from tasks.goal5_segmentation.spatial import restore_binary_to_source
from tasks.results import Goal5Result

#: core / flair 的来源模态优先级（第一个命中者作为掩膜的参考空间）
CORE_SOURCE_MODALITIES = ("t1c", "t1")
FLAIR_SOURCE_MODALITIES = ("flair", "t2")


class Goal5Task(StudyTask[Goal5Result]):
    """T1 增强核心区 + FLAIR/T2 周围总异常区的二值分割。"""

    name = "goal5_segmentation"

    def __init__(self, cfg: Goal5Config | None = None, settings: Settings | None = None,
                 device: str = "cuda") -> None:
        self.cfg = cfg or Goal5Config()
        self.settings = settings or Settings.from_env()
        self.device = device
        self._loaded: LoadedGoal5 | None = None

    # ------------------------------------------------------------------ #
    def load_model(self) -> None:
        """服务启动时加载一次权重（禁止在逐 Study 推理里重复读盘）。"""
        self._loaded = load_model(self.cfg, self.settings.ckpt_root, self.device)

    # ------------------------------------------------------------------ #
    def predict(self, context) -> Goal5Result:
        """单个检查的分割推理。"""
        if self._loaded is None:
            self.load_model()
        assert self._loaded is not None

        study = context.study
        prepared: PreparedVolume = build_volume(study, self.cfg)
        if prepared.missing:
            context.warnings.append(
                f"goal5: 缺通道 {list(prepared.missing)}（已零占位）"
            )

        prob = infer_segmentation(self._loaded, prepared.volume, self.cfg)
        core_p, flair_p = prob[self.cfg.core_channel], prob[self.cfg.flair_channel]

        cfg_pp = self.cfg
        # 阈值来自权重（训练后标定），未标定时退化为配置默认值。
        class _T:                                                # noqa: D401 - 轻量视图
            default_thresholds = self._loaded.thresholds
            min_tumor_voxels = cfg_pp.min_tumor_voxels
            keep_components = cfg_pp.keep_components
            bridge_mm = cfg_pp.bridge_mm

        core_bin, flair_bin = clean_pair(core_p, flair_p, _T())

        # ---- 逆变换：恢复到各自源序列空间（shape/affine 必须与源图一致）----
        core_mask, core_uid = self._restore(study, prepared, core_bin, CORE_SOURCE_MODALITIES)
        flair_mask, flair_uid = self._restore(study, prepared, flair_bin, FLAIR_SOURCE_MODALITIES)

        context.diagnostics["goal5"] = {
            "missing_channels": list(prepared.missing),
            "thresholds": list(self._loaded.thresholds),
            "core_voxels": int(core_mask.sum()),
            "flair_voxels": int(flair_mask.sum()),
            "ckpt": self._loaded.ckpt_path,
        }
        return Goal5Result(
            core_mask=core_mask,
            core_source_series_uid=core_uid,
            flair_mask=flair_mask,
            flair_source_series_uid=flair_uid,
        )

    # ------------------------------------------------------------------ #
    def _restore(self, study, prepared: PreparedVolume, mask: np.ndarray,
                 modalities: tuple[str, ...]) -> tuple[np.ndarray, str]:
        """把公共网格掩膜恢复到指定模态的源序列空间。

        找不到目标模态时退化为参考网格本身（保证仍能写出、shape 自洽），
        并把所用序列 UID 一并返回，供 Writer 决定输出目录。
        """
        picked = select_series(study, modalities)
        src = next(iter(picked.values()), None)
        if src is None:
            return mask.astype(np.uint8), ""

        restored = restore_binary_to_source(
            mask, prepared.affine, np.asarray(src.affine, dtype=np.float64),
            tuple(int(x) for x in src.image.shape),
        )
        return restored.astype(np.uint8), infer_uid_from_path(src)


def build_task(settings: Settings | None = None, **kwargs: Any) -> Goal5Task:
    """工厂入口，供 ``core/registry`` 或团队 Pipeline 构建使用。"""
    return Goal5Task(settings=settings, **kwargs)
