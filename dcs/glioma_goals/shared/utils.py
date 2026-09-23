"""共享工具：配置加载、日志、路径解析。

设计原则：**每个 Goal 的产物互相隔离**。因此这里所有"输出路径"都由
调用方显式传入（通常来自各 Goal 自己的 config.yaml），本模块**不提供全局默认输出目录**——
一旦有了全局默认，两个人不填就会写到同一个地方，日志与权重互相覆盖。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: 数据集根目录（所有 Goal 共享**只读**的数据；可用环境变量覆盖）
ENV_DATASET = "GLIOMA_DATASET_ROOT"

#: 数据根环境变量的**兼容别名**（按顺序取第一个存在的）。
#:
#: 算法工程 ``glioma_track4`` 用的是 ``DATASET_ROOT``，本工程历史上用
#: ``GLIOMA_DATASET_ROOT``。两个工程常常在同一个 shell 里交替跑，只认一个名字会
#: 让人"明明 export 了却没生效"，然后退到 ``../data``（可能根本不存在，
#: 也可能存在但装的是别的数据集——后者更危险）。
ENV_DATASET_ALIASES = (ENV_DATASET, "DATASET_ROOT", "DATASET_PATH")

#: 各 Goal 的缓存根（必须彼此隔离：缓存写入冲突会静默产生半截文件）
ENV_CACHE = "GLIOMA_CACHE_ROOT"


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dataset_root(cli_value: str | None = None) -> Path:
    """数据根目录：CLI > 环境变量（见 :data:`ENV_DATASET_ALIASES`）> ``../data``。"""
    for cand in (cli_value,
                 *(os.environ.get(name) for name in ENV_DATASET_ALIASES),
                 "../data"):
        if cand:
            p = Path(cand).expanduser()
            if p.exists():
                return p.resolve()
    raise FileNotFoundError(
        "找不到数据集根目录；请用 --data 指定，或设置 "
        + " / ".join(ENV_DATASET_ALIASES))


@dataclass
class RunPaths:
    """一个 Goal 一次运行的全部输出路径（**全部落在该 Goal 目录内**）。"""

    root: Path
    ckpt_dir: Path
    cache_dir: Path
    log_dir: Path

    @classmethod
    def for_goal(cls, goal_dir: Path, tag: str) -> "RunPaths":
        root = (goal_dir / "runs" / tag).resolve()
        return cls(root=root, ckpt_dir=root / "checkpoints",
                   cache_dir=(goal_dir / "cache").resolve(), log_dir=root / "logs")

    def ensure(self) -> "RunPaths":
        for d in (self.ckpt_dir, self.cache_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """同时输出到 stdout 与文件的结构化日志。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def append_jsonl(path: Path, row: dict) -> None:
    """训练历史落盘（每行一个 JSON，便于并行运行时各自追加自己的文件）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Timer:
    """上下文计时的轻量实现。"""

    def __init__(self) -> None:
        self.t0 = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.ms = int((time.time() - self.t0) * 1000)

    @property
    def seconds(self) -> float:
        return time.time() - self.t0
