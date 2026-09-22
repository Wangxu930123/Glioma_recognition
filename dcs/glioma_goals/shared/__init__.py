"""共享库（**只读**）。

⚠️ 约定：本目录由组长/集成人员单点维护，**五名队员不要修改这里**——
否则一个人的改动会同时影响其他人的训练，违背"互不影响"的目标。
队员需要定制的内容（预处理、增强、损失、评估）请在自己 goal 目录下的
同名文件中覆盖，那里是独立副本。

包含：
    backbone_mednext / backbone_unet3d   骨干网络
    selector                             序列选择（模态识别）
    volume                               多通道体积预处理（1mm 公共网格）
    spatial                              世界坐标重采样与逆变换
    sliding                              滑窗 + TTA 推理
    data                                 数据集与增强
    engine                               通用训练循环（EMA/AMP/warmup）
    metrics                              分割与重复影像指标
    factory                              按 checkpoint 元信息重建网络
    utils                                配置/日志/路径
"""
from __future__ import annotations

__all__ = ["backbone_mednext", "backbone_unet3d", "selector", "volume", "spatial",
           "sliding", "data", "engine", "metrics", "factory", "utils"]
