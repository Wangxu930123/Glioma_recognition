"""Goal3 Tumor 的模型定义（独立副本，可自由修改）。

默认复用 ``shared`` 中的骨干；本文件的价值在于声明**本任务用哪一路输出**，
以及允许你换成完全不同的网络而不影响其他 Goal。
"""
from __future__ import annotations

from shared.backbone_mednext import MedNeXtNet
from shared.backbone_unet3d import GliomaNet

#: 本任务读取的骨干输出（供 losses.py 与 evaluate.py 参照）
HEAD = 'cls[TumorProbability]'

#: 本任务的分类头定义（仅 goal3_tumor 使用；多任务骨干会同时构建全部头，
#: 但训练时只对本任务那一路回传损失）
CLS_SPEC: list[tuple[str, int]] = [('TumorProbability', 1)]


def build_model(cfg: dict):
    """按 config.yaml 构建骨干（结构与 ``shared.factory`` 保持一致，便于互相加载）。"""
    mc = cfg.get("model") or {}
    in_ch = int(mc.get("in_channels", 4))
    spec = CLS_SPEC
    if str(mc.get("arch", "mednext")) == "resunet":
        return GliomaNet(in_ch=in_ch, base=int(mc.get("base", 32)), cls_spec=spec)
    return MedNeXtNet(
        in_ch=in_ch, base=int(mc.get("base", 32)), depth=int(mc.get("depth", 4)),
        cls_spec=spec, blocks_per_stage=int(mc.get("blocks_per_stage", 2)),
        k=int(mc.get("k", 3)), expand=int(mc.get("expand", 2)),
        aniso_z=bool(mc.get("aniso_z", False)), max_ch=int(mc.get("max_ch", 320)),
    )
