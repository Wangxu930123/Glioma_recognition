"""赛道四自建模型组：`glioma_track4` 算法工程的接入点。

本包**只做路径注入与转发**，全部算法实现位于同级的独立算法工程
``glioma_track4``（训练 + 推理 + 桥接层），因此：

- 比赛协议、Writer/Validator/Aggregator 仍由本仓库单点维护；
- 算法工程可独立训练、独立演进，不侵入公共框架；
- 算法产物（checkpoint）按规范放在
  ``/2026aicompetition/workspace/checkpoint/<goal>/``。

接入方式（见 `configs/competition.env.example`）::

    COMPETITION_PIPELINE_FACTORY=tasks.glioma.pipeline:build_pipeline

部署布局（两者同级）::

    /2026aicompetition/workspace/
    ├── Glioma_recognition/     # 本仓库（提交入口）
    └── glioma_track4/          # 算法工程
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_track4_on_path() -> Path | None:
    """把算法工程加入 ``sys.path``；返回其根目录（找不到返回 None）。"""
    here = Path(__file__).resolve()
    candidates = [
        os.environ.get("GLIOMA_TRACK4_ROOT"),
        here.parents[3] / "glioma_track4",                        # 同级目录（开发期）
        Path("/2026aicompetition/workspace/glioma_track4"),       # 平台约定目录
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "integration" / "factory.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    return None


TRACK4_ROOT = _ensure_track4_on_path()

__all__ = ["TRACK4_ROOT"]
