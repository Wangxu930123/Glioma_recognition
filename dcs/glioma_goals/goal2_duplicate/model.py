"""目标二-B 重复影像检测的模型定义（独立副本，可自由修改）。

本任务用的是骨干的 **embed 头**（L2 归一化嵌入），训练目标是让
"同一检查的两次增强"靠近、不同检查的样本远离。
"""
from __future__ import annotations

from shared.backbone_mednext import MedNeXtNet
from shared.backbone_unet3d import GliomaNet

#: 本任务读取的骨干输出
HEAD = "embed"

#: 分类头定义（多任务骨干会同时构建，但本任务只用 embed）
CLS_SPEC: list[tuple[str, int]] = [("TumorProbability", 1)]


def build_model(cfg: dict):
    """按 config.yaml 构建骨干。"""
    mc = cfg.get("model") or {}
    in_ch = int(mc.get("in_channels", 4))
    if str(mc.get("arch", "mednext")) == "resunet":
        return GliomaNet(in_ch=in_ch, base=int(mc.get("base", 32)), cls_spec=CLS_SPEC)
    return MedNeXtNet(
        in_ch=in_ch, base=int(mc.get("base", 32)), depth=int(mc.get("depth", 4)),
        cls_spec=CLS_SPEC, blocks_per_stage=int(mc.get("blocks_per_stage", 2)),
        k=int(mc.get("k", 3)), expand=int(mc.get("expand", 2)),
        aniso_z=bool(mc.get("aniso_z", False)), max_ch=int(mc.get("max_ch", 320)),
    )
