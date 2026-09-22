"""配置与路径工具。

设计要点：
- 平台路径（WORKSPACE）与数据根（DATASET_ROOT）支持环境变量覆盖，便于在
  云桌面 / 训推平台容器 / 本地三处运行同一份代码；
- 训练日志目录按《赛事开发规范》要求自动切到 ``{WORKSPACE}/logs``（否则可能被标记为可疑对象）。
"""
from __future__ import annotations

import os
from typing import Any

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

#: 平台私有存储（《训推平台使用指南》）；容器内被自动挂载
WORKSPACE = os.environ.get("WORKSPACE", "/2026aicompetition/workspace")
#: 《赛事开发规范》强制的训练日志目录
PLATFORM_LOGS_DIR = os.path.join(WORKSPACE, "logs")


def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(name: str) -> Any:
    """加载 configs/<name>（name 可带 .yaml 后缀）。"""
    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"
    return load_yaml(os.path.join(CONFIG_DIR, name))


def resolve(path: str) -> str:
    """相对路径按项目根解析；绝对路径原样返回。"""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


#: 官方数据挂载前缀（《训推平台使用指南》）
OFFICIAL_PREFIXES = ("/2026aicompetition/datasets", "/2026aicompetition/public_")


def dataset_root() -> str:
    """当前使用的数据根（``DATASET_ROOT`` 优先，其次 paths.yaml 的 raw.track4）。"""
    if os.environ.get("DATASET_ROOT"):
        return os.environ["DATASET_ROOT"]
    try:
        return str((load_config("paths.yaml").get("raw") or {}).get("track4", ""))
    except Exception:                                             # noqa: BLE001
        return ""


def is_official_data(root: str | None = None) -> bool:
    """数据是否来自大赛官方挂载目录。"""
    r = os.path.abspath(root or dataset_root() or "")
    return any(r.startswith(p) for p in OFFICIAL_PREFIXES)


def data_source_tag(root: str | None = None, phase: str = "train") -> str:
    """按《赛事开发规范》生成 ``data_source`` 日志标识。

    **重要**：早期实现把该字段硬编码为 ``official/train_v1``；若用本地/公开数据
    训练也记成 official，属于**日志不实**。现在按数据根自动区分：

    - 官方挂载路径 → ``official/<phase>_v1``
    - 其他路径     → ``local/<目录名>/<phase>``
    """
    r = os.path.abspath(root or dataset_root() or "")
    if is_official_data(r):
        return f"official/{phase}_v1"
    name = os.path.basename(r.rstrip("/")) or "unknown"
    return f"local/{name}/{phase}"


def assert_data_source(man: dict, phase: str = "train", strict: bool = True) -> str:
    """校验清单（manifest）的数据源与本机当前数据源是否同类。

    **合规闸门**：赛事规则要求"比赛数据只能在大赛专属环境中使用"，
    且日志必须可追溯数据来源。若拿本地/公开数据（如 BraTS）生成的清单
    去训练官方数据（或反之），结果与日志都会不可追溯 —— 因此默认**拒绝执行**。

    返回当前数据源标识；不一致时抛 ``RuntimeError``（``strict=False`` 时仅告警）。
    """
    man_tag = str((man or {}).get("data_source") or "")
    cur_tag = data_source_tag(phase=phase)
    if not man_tag:
        print("[guard] ⚠️ 清单缺少 data_source 字段（可能是旧清单）；"
              "建议重新执行 bash scripts/01_probe.sh")
        return cur_tag
    if man_tag.split("/")[0] != cur_tag.split("/")[0]:
        msg = (f"数据源不一致：清单={man_tag}（{man.get('data_root')}）  "
               f"当前={cur_tag}（{dataset_root()}）。\n"
               "  · 若要用**官方数据**训练：确认 DATASET_ROOT 指向大赛挂载目录，然后\n"
               "    bash scripts/01_probe.sh && bash scripts/02_build_dataset.sh\n"
               "  · 若只是**本地验证**：请使用独立的 CACHE_DIR / CKPT_DIR，不要与正式产物混用")
        if strict:
            raise RuntimeError(msg)
        print(f"[guard] ⚠️ {msg}")
    elif man_tag != cur_tag:
        print(f"[guard] ⚠️ 数据源标识变化：清单={man_tag} 当前={cur_tag}（将继续，请确认）")
    return cur_tag


def load_paths() -> dict:
    """路径配置 + 环境变量覆盖 + 平台合规兜底。"""
    p = load_config("paths.yaml")

    ws = os.environ.get("WORKSPACE") or p.get("workspace") or WORKSPACE
    p["workspace"] = ws

    # 数据根：DATASET_ROOT 优先（云桌面里换路径最省事）
    if os.environ.get("DATASET_ROOT"):
        p.setdefault("raw", {})["track4"] = os.environ["DATASET_ROOT"]

    # 训练日志：规范要求写在 {workspace}/logs；平台目录存在时强制切换
    if os.environ.get("LOGS_DIR"):
        p["logs_dir"] = os.environ["LOGS_DIR"]
    elif os.path.isdir(os.path.join(ws, "logs")) or os.path.isdir(ws):
        p["logs_dir"] = os.path.join(ws, "logs")

    # 答案目录：默认 {workspace}/answer（不可写时由 writer 回退到项目内 answer/）
    if os.environ.get("ANSWER_ROOT"):
        p["answer_root"] = os.environ["ANSWER_ROOT"]
    elif os.path.isdir(os.path.join(ws, "answer")):
        p["answer_root"] = os.path.join(ws, "answer")

    for k, env in (("uif_root", "UIF_DIR"), ("manifest", "MANIFEST"),
                   ("folds", "FOLDS"), ("checkpoints_dir", "CKPT_DIR"),
                   ("preprocessed_root", "PREPROCESSED_ROOT"),
                   ("preprocess_cache", "CACHE_DIR")):
        if os.environ.get(env):
            p[k] = os.environ[env]
    return p
