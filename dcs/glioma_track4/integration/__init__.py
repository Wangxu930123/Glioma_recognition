"""与团队串联提交工程（``Glioma_recognition``）的桥接包。

本包把 ``glioma_track4`` 的算法能力封装成团队契约要求的 ``StudyTask`` /
``DatasetTask`` 插件，使本工程能直接挂到团队的比赛 Pipeline 上跑完整提交，
而**不重复实现**协议层（HTTP、Writer、Validator、回调）。

用法：
    export COMPETITION_PIPELINE_FACTORY=tasks.glioma.pipeline:build_pipeline

详细说明见 ``docs/INTEGRATION_WITH_TEAM.md``。
"""

from .factory import build_pipeline

__all__ = ["build_pipeline"]
