"""共享数据管线：读病例 → 1mm 公共网格 → patch 裁剪 → 增强。

设计目标是"**各 Goal 都能复用，但标签各不相同**"：

- 本模块负责**与任务无关**的部分：发现 NIfTI、重采样到公共网格、切 patch、增强；
- 各 Goal 的 ``dataset.py`` 继承 :class:`BaseCaseDataset`，只实现 ``build_label()``
  来产出自己的监督信号（检查级标签 / 配对 / 14 字段 / 分割掩膜）。

为什么把增强也放在共享库：增强策略会显著影响指标，如果每人各改一套，
实验结果就无法横向比较（"我的 Dice 高"可能只是因为增强更弱、
验证集更"干净"）。队员若确实需要不同增强，在**自己目录**里覆盖
``augment`` 即可，但请在 README 里写明，便于复盘。

**缓存隔离**：``cache_dir`` 由调用方传入（各 Goal 自己的目录）。
共享缓存看似省磁盘，但两个进程同时写同一个缓存文件会产生半截文件，
且这类损坏不会报错、只会让指标莫名变差。
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.selector import guess_modality, pick_series
from shared.volume import CHANNEL_ORDER, build_volume

#: 影像文件后缀
NIFTI_SUFFIXES = (".nii", ".nii.gz")

#: 文件名含这些词的视为**掩码/标注**，不作输入影像。
#: ⚠️ 中文关键词必不可少：本数据集用 ``瘤体.nii.gz`` / ``水肿.nii.gz`` /
#: ``肿瘤瘤体.nii.gz`` 命名掩码，只用英文关键词会把掩码当成影像读进来，
#: 后果是"输入通道被污染 + 没有任何监督信号"（dice 恒为 0），且不报错。
MASK_HINTS = ("mask", "seg", "label", "roi", "掩码", "标注",
              "瘤体", "水肿", "异常", "核心", "病灶", "肿瘤区")

#: 平台 ``/2026aicompetition/datasets`` 下的阶段目录名。
#: 数据根必须精确到其中一个（训练用 ``training``），不能停在父目录。
_PLATFORM_PHASES = frozenset({
    "training", "evaluation_first", "evaluation_second",
    "evaluation_finals", "verification",
})


def assert_case_root(root: Path) -> None:
    """拦截"数据根误指向 ``datasets/`` 父目录"。

    平台上 ``/2026aicompetition/datasets`` 下是 5 个阶段目录，数据根要精确到
    ``.../datasets/training``。误传父目录时旧实现会把阶段名当成病例号：

    * 训练照常启动、损失照常下降，但输入是 5 个"检查"混合的像素；
    * 金标准一张也对不上（``no_labels`` 全空），分类/分割都学不到东西；
    * 由于不报错，往往要等到提交或人工核对时才暴露 —— 白烧几小时 GPU。

    因此这里直接失败，并把"应该填哪个路径"写进报错信息。
    """
    root = Path(root)
    if not root.is_dir():
        return
    children = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(children) < 2 or not {c.casefold() for c in children} <= _PLATFORM_PHASES:
        return
    raise ValueError(
        f"数据根 {root} 指向数据集父目录，其下是平台阶段目录 {children}。"
        f"请把数据根设为具体阶段（训练应为 {root / 'training'}）；"
        f"否则这些目录名会被当作病例号，训练数据完全错误却不报错。"
    )


# --------------------------------------------------------------------------- #
# 序列类型解析（官方数据的模态**只能**从这里来）
# --------------------------------------------------------------------------- #
#: sidecar JSON 里可能承载序列类型的键（按优先级）
_SIDECAR_DESC_KEYS = ("SeriesType", "series_type", "SeriesDescription",
                      "ProtocolName", "SequenceName", "modality", "Modality")

#: 掩膜的"核心区"关键词（角色判定与 ``find_masks`` 同一套语义）
_ROLE_CORE_KW = ("core", "核心", "瘤体", "增强", "et", "肿瘤", "tumor")
#: 掩膜的"周围总异常区"关键词
_ROLE_PERI_KW = ("peri", "perimeter", "水肿", "异常", "whole", "总")

#: 已就"缺 openpyxl"告警过（避免每条病例刷一次屏）
_WARNED_NO_OPENPYXL = False


def _norm_key(value) -> str:
    """归一化用于查表的键（去空白 + 大小写无关），与提交工程同一规则。"""
    return re.sub(r"\s+", "", str(value if value is not None else "")).casefold()


def read_series_types(root: Path) -> dict[tuple[str, str], str]:
    """读序列类型表（``3_serieslabel.xlsx`` / ``SeriesType.xlsx`` 两个命名）：
    ``(检查号, 序列号) → 序列类型``。

    ⚠️ **数据里这是模态的唯一来源**。序列目录名是 DICOM UID
    （``2.25.135...``），任何"按名字猜模态"的关键词都命中不了，
    于是出现"扫出几千例、却一例都没有可用序列"——但病例计数看起来完全正常，
    很容易被误判成数据损坏或路径写错。

    **两个命名、两个来源，数据集那份优先**（列都是 检查号 + 序列号 + 类型）：

    * ``SeriesType.xlsx`` —— **赛道四数据集的内容**：``<阶段>/annotation/`` 下、
      与病例目录同层（表头实测就是 ``AccessionNumber`` / ``SeriesUid`` /
      ``SeriesType``；取值 **5 类**：``T1`` / ``T1CE（增强）`` / ``T2-Flair`` /
      ``T2WI`` / ``其他``）。训练集、验证集都有，**评测集随测试数据一起下发**；
    * ``3_serieslabel.xlsx`` —— 团队工作区 ``labels/``（**不是数据集的内容**，
      属另一个目标的产物；仅作兜底、不覆盖数据集取值）。

    定位见 :func:`shared.official_labels.label_search_dirs`：显式/环境变量 →
    ``<工程>/labels`` → **``$WORKSPACE`` 下 3 层** → 数据根/父/祖父 →
    这些目录下像标注容器的一级子目录（``annotation`` / ``标注结果`` …）。
    只认其中一个名字、或只认数据根那一层，都会在另一半环境里翻车。
    表里检查号列与磁盘目录名对不上**不再是问题**：查表走两级口径
    （精确键 → SeriesUid 单键回退，见 :func:`shared.official_labels.lookup_series_type`）。

    与提交工程 ``data/metadata.py: read_series_types`` 保持同一语义：
    表头别名容错、缺文件返回空表（不是错误）、同一键冲突取值**直接失败**。

    找不到 openpyxl 时给出**一次性显式告警**：静默返回空表会让人去改
    真正没错的地方（数据布局），而问题其实只是缺个依赖。
    """
    global _WARNED_NO_OPENPYXL

    from shared.official_labels import (find_named_table, find_official_labels,
                                        read_series_labels)

    out: dict[tuple[str, str], str] = {}

    # ---- ① `SeriesType.xlsx`（**赛道四数据集里的就是这张**，权威取值）----
    # 位置与影像同层：`<阶段>/annotation/SeriesType.xlsx`（训练/验证集已下发，
    # 评测集在正式测试时随测试数据一起下发）。早期这里写的是
    # `Path(root) / "SeriesType.xlsx"`：只认数据根那一层，于是数据根指成
    # annotation/ 的上一层（或表被放进容器子目录）时，表就在磁盘上却读不到 ——
    # 现在与官方表走同一批候选目录（数据根/父/祖父 + 像标注容器的子目录）。
    hit = find_named_table("SeriesType.xlsx", root)
    path = Path(hit) if hit else None
    if path is None:
        return _read_legacy_series_types(out, root)
    try:
        from openpyxl import load_workbook
    except ImportError:                                           # pragma: no cover
        if not _WARNED_NO_OPENPYXL:
            _WARNED_NO_OPENPYXL = True
            print(f"[data][告警] 发现 {path} 但未安装 openpyxl，序列类型读不到 → "
                  f"UID 命名的序列会全部认不出模态。请先 pip install openpyxl",
                  flush=True)
        return _read_legacy_series_types(out, root)

    aliases = {
        "acc": ("accessionnumber", "accession", "检查号", "检查编号", "病例号"),
        "uid": ("seriesinstanceuid", "seriesuid", "序列号", "序列uid"),
        "typ": ("seriestype", "type", "序列类型", "模态", "序列描述"),
    }
    workbook = load_workbook(path, read_only=True, data_only=True)
    seen_here: dict[tuple[str, str], str] = {}                     # 仅用于本文件内冲突检测
    try:
        for sheet in workbook.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            header = [_norm_key(c) for c in rows[0]]
            idx: dict[str, int] = {}
            for want, keys in aliases.items():
                for i, h in enumerate(header):
                    if any(k in h for k in keys):
                        idx[want] = i
                        break
            if set(idx) != {"acc", "uid", "typ"}:
                continue                                          # 该 sheet 不是映射表
            for row in rows[1:]:
                try:
                    acc, uid, typ = row[idx["acc"]], row[idx["uid"]], row[idx["typ"]]
                except IndexError:
                    continue
                if acc is None or uid is None or typ is None:
                    continue
                key = (_norm_key(acc), _norm_key(uid))
                value = str(typ).strip()
                if not value:
                    continue
                if key in seen_here and seen_here[key] != value:
                    raise ValueError(
                        f"SeriesType.xlsx 冲突：检查号={acc!r} 序列={uid!r} "
                        f"同时映射到 {seen_here[key]!r} 与 {value!r}（{path}）")
                seen_here[key] = value
                out[key] = value                                  # 数据集那份 = 权威取值
    finally:
        workbook.close()
    if seen_here:
        print(f"[data] 已读序列类型表 {path.name}：{len(seen_here)} 条（{path}）", flush=True)
    return _read_legacy_series_types(out, root)


def _read_legacy_series_types(out: dict[tuple[str, str], str],
                              root: Path) -> dict[tuple[str, str], str]:
    """补读工作区那份 ``3_serieslabel.xlsx``：**只补缺，不覆盖** ``out``。

    它**不是赛道四数据集的内容**（属工作区里另一个目标的产物），列结构恰好同构
    （``SeriesLabel ∈ {T1CE,T2,FLAIR}``），所以留作兜底；数据集里的
    ``SeriesType.xlsx`` 一旦给出同一个 ``(检查号, 序列号)``，就以数据集为准。
    顺序反了会**静默覆盖**权威取值（工作区那份取值更粗，如只写 ``T2``，
    而数据集分 ``T2WI``/``T2-Flair``），表现是"模态看着都认出来了、通道里却是错的对比度"。
    """
    from shared.official_labels import find_official_labels, read_series_labels

    series_file = find_official_labels(root).get("series")
    if not series_file:
        return out
    added = 0
    for (acc, uid), value in read_series_labels(series_file).items():
        key = (_norm_key(acc), _norm_key(uid))
        if key not in out:
            out[key] = value
            added += 1
    if added:
        print(f"[data] 已读兼容类型表 {Path(series_file).name}：{added} 条"
              f"（{series_file}；数据集里的 SeriesType.xlsx 优先）", flush=True)
    return out


def _sidecar_desc(path: Path) -> str | None:
    """读同名 JSON sidecar 的序列描述（不存在或解析失败返回 None）。"""
    stem = path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem
    sidecar = path.with_name(stem + ".json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:                                             # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    for key in _SIDECAR_DESC_KEYS:
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _series_desc(path: Path, accession: str, uid: str, series_types: dict,
                 uid_index: dict | None = None) -> str:
    """序列类型的**取用优先级**：类型表 → sidecar → 目录名。

    返回的文本会作为 ``Series`` 的 ``modality`` 交给模态关键词匹配，
    因此它可以是 ``T1CE`` / ``FLAIR`` / ``T1增强`` 这类**任意自然描述**。

    查表时**目录名与文件名都试**：官方数据的组织方式不止一种 ——
    ``<检查号>/<序列号>/<序列号>.nii.gz``（序列号在目录名上）
    与 ``<检查号>/<序列号>.nii.gz``（序列号在文件名上）都可能出现，
    只试目录名会让后者整批认不出模态。

    查表走两级（:func:`shared.official_labels.lookup_series_type`）：先
    ``(检查号, 序列号)`` 精确键，查不到再按 **SeriesUid 单键回退** ——
    表里的检查号列与磁盘目录名口径不一致时（匿名化/哈希），精确键会整批落空。
    """
    from shared.official_labels import lookup_series_type

    stem = path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem
    value = lookup_series_type(series_types, accession, (uid, stem), uid_index)
    if value:
        return str(value)
    value = _sidecar_desc(path)
    if value:
        return value
    return uid


def _mask_role(filename: str, desc: str) -> str | None:
    """由**文件名 + 序列类型**判定掩膜角色；``None`` 表示这是影像。

    角色判定必须结合所在序列的模态：FLAIR/T2 上的"瘤体"属于**总异常区**，
    而不是核心区——只看文件名会把两者的空间搞混。
    """
    text = f"{filename} {desc}".casefold()
    if not any(h in text for h in MASK_HINTS):
        return None
    is_flair_like = any(k in desc.casefold() for k in ("flair", "t2"))
    core_like = any(k in text for k in _ROLE_CORE_KW)
    peri_like = any(k in text for k in _ROLE_PERI_KW)
    if peri_like and not core_like:
        return "peri"
    if core_like:
        return "peri" if is_flair_like else "core"
    if peri_like:
        return "peri"
    return "core"                                                 # 仅命中"标注/掩码"等泛化词


# --------------------------------------------------------------------------- #
# 病例发现
# --------------------------------------------------------------------------- #
#: 官方数据里"异常影像"的子目录名（天坛参考实现 ``paths.py``）：
#: ``<根>/fake/``、``<根>/compositing/``、``<根>/duplicate/``，与主目录**同构**。
#: 绝不能被当成检查号（否则会出现三个名叫 fake/compositing/duplicate 的"病例"），
#: 但它们的影像**正是目标一/二的正样本**，必须作为带标记的病例并入。
SPECIAL_SOURCE_DIRS = ("fake", "compositing", "composition", "duplicate")

#: 允许自动下钻的中间层：``annotation``（影像/标注表所在层）+ 平台阶段名。
_DESCEND_DIRS = frozenset({"annotation"}) | _PLATFORM_PHASES
#: 顶层非病例目录：本层出现其中任何一个，说明"还没到病例层"
_NON_CASE_DIRS = (frozenset({"annotation", "cache", "runs", "folds", "labels",
                             "logs", "checkpoints", "weights"})
                  | frozenset(SPECIAL_SOURCE_DIRS))


def resolve_case_root(root):
    """把"填高了一层"的数据根下钻到真正含病例目录的那一层。

    平台实测结论见 ``glioma_track4/docs/CLOUD_DESKTOP_RUNBOOK.md`` §3.2：
    ``/2026aicompetition/datasets/training`` 下**只有** ``annotation/``，
    影像与 ``SeriesType.xlsx`` 都在 ``training/annotation/``。

    填高一层不报错、只静默扫到 0 例（训练照常启动、损失照常不动）。
    规则与提交工程 ``data/loader.py::_resolve_dataset_root`` 一致：
    仅当"下一层唯一候选"时下钻并告警，候选多于一个时交给
    :func:`assert_case_root` 报错。
    """
    root = Path(root)
    if not root.is_dir():
        return root
    if any(p.is_dir() and p.name.lower() not in _NON_CASE_DIRS for p in root.iterdir()):
        return root                                       # 本层已经有病例目录
    children = sorted(p for p in root.iterdir() if p.is_dir())
    cands = [p for p in children if p.name.casefold() in _DESCEND_DIRS]
    if len(cands) == 1:
        print(f"[data][告警] 数据根 {root} 下没有病例目录，已自动下钻到 "
              f"{cands[0].name}/（若不对请用 --data / GLIOMA_DATASET_ROOT 指定）",
              flush=True)
        return cands[0]
    return root


def _official_context(root: Path) -> dict:
    """一次性读取官方 5 张标注表（缺失的键为默认空值）。

    这是研发侧**唯一权威**的标签来源：模态、掩膜、结构化字段、异常标记
    全都来自它，而不是 ``label.json``、目录名关键词或中文列名——
    官方数据里那些都不存在，于是 special 标签恒为 0、字段全空，
    训练照常跑完却什么都没学到（最难发现的一类失效）。
    """
    from shared.official_labels import (find_official_labels, read_abnormal_labels,
                                        read_characteristics, read_duplicate_pairs,
                                        read_mask_labels)

    files = find_official_labels(root)
    if files:
        print("[data] 官方标注表：" + ", ".join(
            f"{k}={Path(v).name}" for k, v in files.items()), flush=True)
    mask_by_acc: dict[str, dict[str, list[str]]] = {}
    if files.get("mask"):
        for (acc, uid), names in read_mask_labels(files["mask"]).items():
            slot = mask_by_acc.setdefault(_norm_key(acc), {})
            slot[_norm_key(uid)] = names
    abnormal: dict[tuple[str, str], str] = {}
    if files.get("abnormal"):
        abnormal = {(_norm_key(a), _norm_key(u)): v
                    for (a, u), v in read_abnormal_labels(files["abnormal"]).items()}
    labels: dict[str, dict] = {}
    if files.get("characteristics"):
        labels = read_characteristics(files["characteristics"])
    pairs: list[tuple[str, str]] = []
    if files.get("duplicate"):
        pairs = read_duplicate_pairs(files["duplicate"])
    return {"files": files, "masks": mask_by_acc, "abnormal": abnormal,
            "labels": labels, "pairs": pairs}


def discover_cases(dataset_root: Path, limit: int | None = None) -> list[dict]:
    """扫描 ``<root>/<AccessionNumber>/<SeriesUid>/*.nii[.gz]``。

    返回的每个 case 除 ``accession``/``dir``/``series`` 外，还可能带：

    - ``series[].desc``：序列类型描述（官方 `3_serieslabel.xlsx` → sidecar → 目录名）
    - ``masks``：按角色分组的掩膜（官方 `4_masklabel.xlsx` 的 Maskname 或名称启发）
    - ``labels``：结构化字段（官方 `5_characteristics.xlsx`，已是规范字段名）
    - ``special``：``{fake, stitched, duplicate}``（官方 `1_abnormal.xlsx`）
    - ``source``：``true`` / ``fake`` / ``compositing`` / ``duplicate``（影像所在子目录）

    只做轻量发现（不读体素），真正的读取延迟到 ``load_case``。
    """
    root = resolve_case_root(Path(dataset_root))   # 填高一层（如 .../training）时自动下钻
    assert_case_root(root)
    from shared.official_labels import build_uid_index

    series_types = read_series_types(root)
    # 表里的检查号列与磁盘目录名对不上时，靠它按 SeriesUid 单键回退（见 _series_desc）
    uid_index = build_uid_index(series_types)
    ctx = _official_context(root)
    cases: list[dict] = []

    def _collect(acc_dir: Path, source: str) -> dict | None:
        """扫一个病例目录 → case dict（无可用序列时返回 None）。"""
        series: list[dict] = []
        masks: dict[str, list[str]] = {}
        acc_key = _norm_key(acc_dir.name)
        table_masks = ctx["masks"].get(acc_key, {})
        for f in sorted(acc_dir.rglob("*")):
            if not f.is_file() or not f.name.lower().endswith(NIFTI_SUFFIXES):
                continue
            uid = f.parent.name
            desc = _series_desc(f, acc_dir.name, uid, series_types, uid_index)
            # 官方掩膜表：文件名任意（core.nii.gz…），关键词认不出，必须查表
            if f.name in (table_masks.get(_norm_key(uid)) or []):
                role = _mask_role(f.name, desc) or "core"
                masks.setdefault(role, []).append(str(f))
                continue
            role = _mask_role(f.name, desc)
            if role:
                masks.setdefault(role, []).append(str(f))
                continue
            if any(h in f.name.lower() for h in MASK_HINTS):
                continue
            series.append({"path": str(f), "uid": uid, "desc": desc})
        if not series:
            return None
        case = {"accession": acc_dir.name, "dir": str(acc_dir),
                "series": series, "source": source, "root": str(root)}
        if masks:
            case["masks"] = masks
        rec = ctx["labels"].get(acc_dir.name) or ctx["labels"].get(acc_key)
        if rec:
            case["labels"] = dict(rec)
        return case

    def _special_of(case: dict) -> dict[str, float]:
        """由官方 `1_abnormal.xlsx` 汇总出该病例的特殊影像标记（逐序列 → 取最大）。"""
        from shared.official_labels import special_flags

        flags = {"fake": 0.0, "stitched": 0.0, "duplicate": 0.0}
        for s in case["series"]:
            label = ctx["abnormal"].get((_norm_key(case["accession"]), _norm_key(s["uid"])))
            if label:
                for k, v in special_flags(label).items():
                    flags[k] = max(flags[k], v)
        if case.get("source") == "fake":
            flags["fake"] = 1.0
        elif case.get("source") in ("compositing", "composition"):
            flags["stitched"] = 1.0
        elif case.get("source") == "duplicate":
            flags["duplicate"] = 1.0
        return flags

    # ---- 正常影像（主目录一级子目录 = 检查号）----
    for acc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        low = acc_dir.name.lower()
        if low in ("annotation", "cache", "runs") or low in SPECIAL_SOURCE_DIRS:
            continue
        case = _collect(acc_dir, "true")
        if case:
            case["special"] = _special_of(case)
            cases.append(case)
        if limit and len(cases) >= limit:
            break

    # ---- 异常影像（fake / compositing / duplicate）：与主目录同构，逐例并入 ----
    # 它们是目标一（真实性）与目标二-A（拼接）**唯一的正样本来源**，
    # 漏掉这一段的后果是这两个头永远学不到东西（而且不报错）。
    for src in SPECIAL_SOURCE_DIRS:
        src_dir = root / src
        if not src_dir.is_dir():
            continue
        added = 0
        for acc_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
            case = _collect(acc_dir, src)
            if case:
                case["special"] = _special_of(case)
                cases.append(case)
                added += 1
            if limit and len(cases) >= limit:
                break
        if added:
            print(f"[data] 已并入 {src}/ 下 {added} 例异常影像"
                  f"（目标一/二-A 的正样本来源）", flush=True)

    # 前置预警：一条序列都认不出模态时，训练会在 DataLoader worker 里抛
    # 「无任何可用序列」——堆栈落在 torch 的取数内部，看不出根因。
    if cases and not any(
        guess_modality(s.get("desc") or s.get("uid") or "")
        for c in cases[:50] for s in c.get("series") or []
    ):
        from shared.official_labels import describe_modality_sources

        print("[data] ⚠️ 没有任何序列能识别出模态（前 50 例逐条试过：目录名/文件名不含关键词，"
              "类型表也没给出 T1CE/T2/FLAIR 这类取值）。\n"
              f"       自检：{describe_modality_sources(root)}\n"
              "       继续训练会在取数时报「无任何可用序列」。\n"
              "       处理：先 find $WORKSPACE -name 3_serieslabel.xlsx 定位；表若本来就在，"
              "说明清单/标注有问题（把自检行与报错原文一起贴出来）；确实缺表就 "
              "export GLIOMA_LABELS_DIR=<它所在目录>（或软链到 <工程>/labels）",
              flush=True)
    return cases


def _candidate_folds(cfg: dict, data_root, goal_dir) -> list[Path]:
    """统一折划分（``folds.json``）的候选位置，按优先级排列。"""
    dc = cfg.get("data") or {}
    cands: list[Path] = []
    if dc.get("folds"):
        cands.append(Path(str(dc["folds"])))
    if os.environ.get("GLIOMA_FOLDS"):
        cands.append(Path(os.environ["GLIOMA_FOLDS"]))
    cands.append(Path(data_root) / "folds.json")
    # 推荐布局：算法工程与本训练工程并列，折划分由算法工程统一产出
    cands.append(Path(goal_dir).resolve().parent.parent
                 / "glioma_track4" / "data" / "folds.json")
    return cands


def split_cases(cases: list[dict], val_ratio: float, seed: int):
    """按 accession 稳定划分 train/val（同一 seed 结果可复现）。"""
    idx = list(range(len(cases)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(len(cases) * float(val_ratio))))
    val_idx = set(idx[:n_val])
    return ([c for i, c in enumerate(cases) if i not in val_idx],
            [c for i, c in enumerate(cases) if i in val_idx])


def available_folds(cfg: dict, data_root, goal_dir) -> tuple[Path | None, list[str]]:
    """返回 ``(折划分文件, 可用折号列表)``；没有 ``folds.json`` 时返回 ``(None, [])``。

    单独暴露它，是为了让训练入口能在**开跑前**校验 ``--fold N``：
    折号写错时 :func:`split_train_val` 只会打印一行提示并**退回按比例划分**，
    于是"六个 Goal 用同一折"这个前提被悄悄破坏 —— 指标不可比、权重也无法合并，
    而训练日志、损失曲线一切正常。

    （只读文件、不改任何东西；供命令行入口做参数校验用。）
    """
    for path in _candidate_folds(cfg, data_root, goal_dir):
        try:
            if not Path(path).is_file():
                continue
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data:
            return Path(path), sorted(str(k) for k in data)
    return None, []


def split_train_val(cases: list[dict], cfg: dict, data_root, goal_dir,
                    seed: int = 42) -> tuple[list[dict], list[dict], str]:
    """决定 train/val 划分，返回 ``(train_cases, val_cases, source)``。

    **优先使用统一折划分**（``folds.json``）。这条与"五个 Goal 是由五个人
    并行做、还是一个人串行做"**无关**，两种做法都该用同一份折划分：

    * 一个人串行做完五个 Goal 时，统一口径才让五个指标**互相可比**，
      也才能判断"哪个 Goal 弱、该补哪里"；
    * 五个人并行做时，统一口径才让各 Goal 的权重可以**合并/集成**——
      若各人按各自的 ``val_ratio`` 划分，"目标一的正样本落在哪一折"
      每人都不同，合并后既无法评估也无法复现。

    按 ``val_ratio`` 自行划分只作为**过渡**：在 ``folds.json`` 还不存在时
    不阻塞训练，但日志会明确提示其代价。

    Returns:
        ``(train_cases, val_cases, source)``，``source`` ∈ ``{"folds", "ratio"}``。

    Raises:
        ValueError: 数据量不足以划分出非空验证集。
    """
    tr = cfg.get("train") or {}
    fold = int(tr.get("fold", 0))

    for p in _candidate_folds(cfg, data_root, goal_dir):
        if not p.is_file():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                folds = json.load(f)
        except Exception:                                         # noqa: BLE001
            continue
        split = folds.get(str(fold))
        if not split:
            continue
        tr_acc = set(split.get("train") or [])
        va_acc = set(split.get("val") or [])
        trc = [c for c in cases if c["accession"] in tr_acc]
        vac = [c for c in cases if c["accession"] in va_acc]
        # 划分与当前数据必须**对得上**：换了数据集/子集时静默沿用错位划分，
        # 会让"验证集"里混进训练样本——指标虚高且无从察觉。
        if trc and vac:
            print(f"[data] 验证集 = 统一折划分 {p}（fold={fold}）"
                  f" train={len(trc)} val={len(vac)}", flush=True)
            return trc, vac, "folds"
        print(f"[data] ⚠️ {p} 的 fold={fold} 与当前数据不匹配"
              f"（train={len(trc)} val={len(vac)} —— 病例号对不上）→ 退回 val_ratio 划分。"
              f"最常见原因：该折划分来自**另一个数据根**（如从本地验证集切到官方数据后"
              f"没重建）。请在算法工程目录按当前数据重建："
              f"bash scripts/01_probe.sh && bash scripts/02_build_dataset.sh"
              f"（换数据根后必须先跑 01，否则 02 会把旧清单的折当成本数据的折）",
              flush=True)
        break

    val_ratio = float(tr.get("val_ratio", 0.2))
    trc, vac = split_cases(cases, val_ratio, seed)
    print(f"[data] ⚠️ 未找到可用折划分，按 val_ratio={val_ratio} 自行划分"
          f"（train={len(trc)} val={len(vac)}）。各 Goal 划分不同会让指标"
          f"不可比、权重无法合并，建议尽快产出统一的 folds.json", flush=True)
    return trc, vac, "ratio"


def find_masks(case: dict) -> dict[str, list[str]]:
    """找出该病例的掩码文件，按角色归类为**列表**。

    同一角色可能有多个来源掩码（例如 FLAIR 目录下同时给了
    ``瘤体.nii.gz`` 与 ``水肿.nii.gz``，按任务定义二者**并集**才是
    "周围总异常区"）。若用单值字典，后一个会静默覆盖前一个，
    监督信号就少了一块，而这类缺失不会报错。

    角色判定结合**文件名**与**所在序列的模态**：FLAIR 序列上的"瘤体"属于
    "总异常区"而不是"核心区"——只看文件名会把两者的空间搞混。

    两路结果**取并集**：``case["masks"]`` 来自类型表/sidecar（官方数据的主力，
    文件名是 UID 时只有它认得出来），文件名启发式则覆盖模拟集与零散命名。
    任一单路都可能有遗漏，并集最稳。
    """
    out: dict[str, list[str]] = {
        role: list(paths) for role, paths in (case.get("masks") or {}).items()
    }
    for f in sorted(Path(case["dir"]).rglob("*")):
        if not f.is_file() or not f.name.lower().endswith(NIFTI_SUFFIXES):
            continue
        name = f.name.lower()
        if not any(h in name for h in MASK_HINTS):
            continue
        parent = f.parent.name.lower()
        # 掩码的**任务角色**要结合它所在序列的模态判断：
        # FLAIR/T2 上的"瘤体"属于"总异常区(peri)"，而不是"核心区(core)"——
        # 只看文件名会把两者的空间搞混（掩码写到错误的序列空间上）。
        # 目录名是 UID 时这里判不出模态，改用类型表给出的 ``desc``。
        desc = ""
        for s in case.get("series") or []:
            if s.get("uid") == f.parent.name:
                desc = str(s.get("desc") or "")
                break
        role = _mask_role(f.name, f"{parent} {desc}")
        if role:
            out.setdefault(role, []).append(str(f))
    return {role: sorted(set(paths)) for role, paths in out.items()}


# --------------------------------------------------------------------------- #
# 读盘与缓存
# --------------------------------------------------------------------------- #
def load_nii(path: str | Path) -> np.ndarray:
    """读 NIfTI 体数据（float32）。"""
    import nibabel as nib

    img = nib.load(str(path))
    return np.asanyarray(img.dataobj, dtype=np.float32)


def _case_cache_path(cache_dir: Path, accession: str) -> Path:
    return Path(cache_dir) / f"{accession}.npz"


def load_case(case: dict, common_spacing=(1.0, 1.0, 1.0), cache_dir: Path | None = None):
    """把病例读成公共网格体积（带可选缓存）。

    Returns:
        ``(vol[C,D,H,W], masks{角色: 同网格二值}, meta)``
    """
    cache = _case_cache_path(cache_dir, case["accession"]) if cache_dir else None
    if cache and cache.is_file():
        try:
            z = np.load(cache, allow_pickle=False)
            masks = {k[len("mask_"):]: z[k] for k in z.files if k.startswith("mask_")}
            return z["vol"], masks, json.loads(str(z["meta"]))
        except Exception:                                         # noqa: BLE001
            cache.unlink(missing_ok=True)                         # 半截缓存 → 丢弃重算

    # 延迟导入：只有真正读盘时才需要 nibabel
    series = []
    for s in case["series"]:
        img = _load_with_affine(s["path"])
        series.append({**s, "image": img[0], "affine": img[1]})

    prepared = build_volume(_AsStudy(case["accession"], series,
                                     data_root=case.get("root")), common_spacing)
    vol = prepared.volume

    masks: dict[str, np.ndarray] = {}
    for role, paths in find_masks(case).items():
        merged: np.ndarray | None = None
        for mpath in paths:
            arr, aff = _load_with_affine(mpath)
            if arr.shape != tuple(prepared.shape) or not np.allclose(aff, prepared.affine):
                from shared.spatial import resample_to
                arr = resample_to(arr.astype(np.float32), aff, prepared.shape,
                                  prepared.affine, order=0)
            b = (arr > 0.5)
            # 同角色多掩码取**并集**（如 peri = 瘤体 ∪ 水肿）
            merged = b if merged is None else (merged | b)
        if merged is not None:
            masks[role] = merged.astype(np.uint8)

    meta = {"shape": list(prepared.shape), "spacing": list(common_spacing),
            "missing": list(prepared.missing)}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, vol=vol, meta=json.dumps(meta),
                            **{f"mask_{k}": v for k, v in masks.items()})
    return vol, masks, meta


def _load_with_affine(path: str):
    import nibabel as nib

    img = nib.load(str(path))
    return np.asanyarray(img.dataobj, dtype=np.float32), np.asarray(img.affine, dtype=np.float64)


class _AsStudy:
    """把轻量 dict 适配成 ``build_volume`` 期望的接口（避免共享库绑定团队类）。"""

    def __init__(self, accession: str, series: list[dict],
                 data_root: str | None = None) -> None:
        self.accession_number = accession
        self.series = [_AsSeries(s) for s in series]
        # 供 build_volume 的报错自检定位官方标注表（表不在数据集里，见 DATASET_ROOT 文档）
        self.data_root = data_root


class _AsSeries:
    def __init__(self, s: dict) -> None:
        self.series_uid = s["uid"]
        # 模态取自 ``discover_cases`` 解析出的**类型描述**（类型表 → sidecar → 目录名）。
        # 直接用目录名会让官方的 UID 命名序列全部认不出模态：病例数正常、
        # 但 ``pick_series`` 一个都挑不出来 → "无任何可用序列"。
        self.modality = str(s.get("desc") or s["uid"])
        self.image = s["image"]
        self.affine = s["affine"]
        self.source_path = Path(s["path"])
        # 刻意留空：``selector._key_of`` 把 ``metadata["modality"]`` 当作**权威值**，
        # 若把 sidecar 里的 DICOM 取值（如 "MR"）塞进来，它会把序列判成未知模态
        # 而直接跳过——模态匹配统一走上面的 ``desc``，避免两处口径打架。
        self.metadata = {}


# --------------------------------------------------------------------------- #
# patch 与增强
# --------------------------------------------------------------------------- #
def crop_patch(vol: np.ndarray, patch: tuple[int, int, int], rng: random.Random,
               center: np.ndarray | None = None, pos_ratio: float = 0.7,
               starts: tuple[int, int, int] | None = None):
    """随机（或以病灶为中心）裁剪 patch；体积不足时用 0 填充。

    ``pos_ratio`` 控制"以病灶为中心"的比例：全病灶会让模型只学会看肿瘤、
    全随机又会让正样本比例过低，两者都伤 Dice。

    ``starts``：给定左上角偏移时直接按它裁（**掩码必须与影像用同一 starts**，
    各自独立随机裁剪会让输入和监督信号空间错位，训练出来的模型看似收敛、
    实际学的是错误对应关系）。
    """
    c, d, h, w = vol.shape
    pd, ph, pw = patch
    if center is not None and rng.random() < pos_ratio:
        ctr = np.asarray(center, float)
    else:
        ctr = np.array([d, h, w], float) / 2.0
        if c > 0:
            ctr = ctr + np.array([rng.uniform(-0.25, 0.25) * s for s in (d, h, w)])

    if starts is None:
        starts = []
        for i, (sz, ps) in enumerate(zip((d, h, w), patch)):
            s0 = int(round(ctr[i] - ps / 2.0))
            s0 = max(0, min(s0, max(0, sz - ps)))
            starts.append(s0)
    starts = tuple(int(x) for x in starts)

    out = np.zeros((c,) + tuple(patch), dtype=np.float32)
    sl = tuple(slice(st, st + ps) for st, ps in zip(starts, patch))
    src = vol[(slice(None),) + sl]
    out[:, : src.shape[1], : src.shape[2], : src.shape[3]] = src
    return out, starts


def lesion_center(masks: dict[str, np.ndarray]) -> np.ndarray | None:
    """病灶质心（优先 peri，其次 core）。"""
    for key in ("peri", "core"):
        m = masks.get(key)
        if m is not None and m.any():
            return np.asarray(np.argwhere(m > 0).mean(0), float)
    return None


def augment(vol: np.ndarray, masks: np.ndarray | None, rng: random.Random,
            cfg: dict | None = None):
    """几何 + 强度增强（确定性随机源，可复现）。

    几何变换同时作用于影像与掩码，且掩码用**最近邻**重采样——
    线性插值会在边界造出 0.5 这类中间值，而二值掩膜一旦不纯，
    训练目标就被污染了。
    """
    a = (cfg or {}).get("augment", {}) if cfg else {}
    if not a.get("enabled", True):
        return vol, masks

    # 1) 随机翻转（各轴独立）
    for axis in range(1, 4):
        if rng.random() < float(a.get("flip_prob", 0.5)):
            vol = np.flip(vol, axis).copy()
            if masks is not None:
                masks = np.flip(masks, axis).copy()

    # 2) 随机仿射（小角度旋转 + 缩放 + 平移）
    if masks is not None and rng.random() < float(a.get("affine_prob", 0.3)):
        vol, masks = _random_affine(vol, masks, rng,
                                    max_rot=float(a.get("max_rot_deg", 12.0)),
                                    max_scale=float(a.get("max_scale", 0.12)))

    # 3) 强度扰动（gamma / 线性 + 噪声）——只作用于影像
    if rng.random() < float(a.get("intensity_prob", 0.8)):
        g = 1.0 + rng.uniform(-1, 1) * float(a.get("gamma_range", 0.25))
        s = 1.0 + rng.uniform(-1, 1) * float(a.get("scale_range", 0.15))
        b = rng.uniform(-1, 1) * float(a.get("shift_range", 0.15))
        vol = np.sign(vol) * np.power(np.abs(vol) + 1e-6, g) * s + b
    if rng.random() < float(a.get("noise_prob", 0.3)):
        nrng = np.random.default_rng(rng.getrandbits(32))
        vol = vol + nrng.standard_normal(vol.shape).astype(np.float32) \
            * rng.uniform(0.0, 0.08)
    return vol.astype(np.float32), masks


def _random_affine(vol: np.ndarray, masks: np.ndarray, rng: random.Random,
                   max_rot: float, max_scale: float):
    """小角度旋转 + 缩放（对影像线性插值、对掩码最近邻）。"""
    from scipy import ndimage

    shape = vol.shape[1:]
    ang = [np.deg2rad(rng.uniform(-max_rot, max_rot)) for _ in range(3)]
    sc = [1.0 + rng.uniform(-max_scale, max_scale) for _ in range(3)]

    def _rot(axis, a):
        c, s = np.cos(a), np.sin(a)
        m = np.eye(3)
        i, j = [x for x in range(3) if x != axis]
        m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
        return m

    m = _rot(0, ang[0]) @ _rot(1, ang[1]) @ _rot(2, ang[2])
    m = m @ np.diag(sc)
    off = (np.array(shape, float) - m @ np.array(shape, float)) / 2.0

    # scipy 要求变换矩阵的维度与数组维度一致：
    # 影像是 (C,D,H,W) 4D，掩码可能是 (K,D,H,W) 或 (D,H,W)，
    # 通道维不参与空间变换，因此按 4D 补全矩阵（否则报 "affine matrix has wrong number of rows"）。
    vol = _apply_affine(vol, m, off, order=1)
    masks = (_apply_affine(masks.astype(np.float32), m, off, order=0) > 0.5).astype(np.uint8)
    return vol.astype(np.float32), masks


def _apply_affine(arr: np.ndarray, m: np.ndarray, off: np.ndarray, order: int) -> np.ndarray:
    """把 3D 空间变换应用到 3D 或 4D 数组（通道维保持恒等）。"""
    from scipy import ndimage

    if arr.ndim == 3:
        return ndimage.affine_transform(arr, m, offset=off, output_shape=arr.shape, order=order)
    m4 = np.eye(arr.ndim)
    m4[:3, :3] = m
    off4 = np.zeros(arr.ndim)
    off4[:3] = off
    return ndimage.affine_transform(arr, m4, offset=off4, output_shape=arr.shape, order=order)


# --------------------------------------------------------------------------- #
# 基础数据集
# --------------------------------------------------------------------------- #
class BaseCaseDataset:
    """所有 Goal 数据集的基类：负责读盘/裁剪/增强，子类只管标签。

    子类实现 :meth:`build_label`，返回该任务需要的监督信号字典。
    """

    def __init__(self, cases: list[dict], patch=(96, 96, 96), train: bool = True,
                 common_spacing=(1.0, 1.0, 1.0), cache_dir: Path | None = None,
                 aug_cfg: dict | None = None, seed: int = 42,
                 pos_ratio: float = 0.7,
                 global_view_cfg: dict | None = None) -> None:
        self.cases = cases
        self.patch = tuple(patch)
        self.train = train
        self.common_spacing = common_spacing
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.aug_cfg = aug_cfg or {}
        self.rng = random.Random(seed)
        self.pos_ratio = pos_ratio
        #: 整脑视图配置（供 ``special`` / ``cls`` / ``embed`` 这些"全局头"使用）。
        #:
        #: ⚠️ **训练与推理的物理视野必须一致**。推理侧
        #: ``tasks/_common/volume.global_view`` 固定用 ``size_mm=192 → out=96``
        #: （等效 2mm/体素、覆盖整脑）。早期实现让全局头直接在 96³ patch
        #: （1mm/体素、仅 96mm 视野）上训练，两者视野相差一倍、中心也不同，
        #: 于是"整检查是否拼接/是否假人体"这类判断在推理时分布漂移，
        #: 表现为训练 AUC 很高、上线后接近随机。
        #:
        #: ``enabled=False`` 可关闭（仅用于消融对比，正常训练不要关）。
        #:
        #: 取值优先级：显式参数 > ``config.yaml`` 的 ``global_view`` 段 >
        #: 空字典（即按默认开启）。这样各 Goal 不必改 dataset.py，
        #: 只要在配置里写 ``global_view: {size_mm: 192, out: 96}`` 即可微调。
        self.gv = dict(global_view_cfg if global_view_cfg is not None
                       else (aug_cfg or {}).get("global_view") or {})

    def __len__(self) -> int:
        return len(self.cases)

    # ---- 子类实现 ----
    def build_label(self, case: dict, masks: dict, vol_shape) -> dict:
        """返回该任务的监督信号（键名由子类约定）。"""
        raise NotImplementedError

    # ---- 框架 ----
    def __getitem__(self, i: int) -> dict:
        case = self.cases[i % len(self.cases)]
        vol, masks, _meta = load_case(case, self.common_spacing, self.cache_dir)

        center = lesion_center(masks) if self.train else None
        patch_vol, starts = crop_patch(vol, self.patch, self.rng, center,
                                       self.pos_ratio if self.train else 0.0)

        patch_masks: dict[str, np.ndarray] = {}
        if masks:
            for role, m in masks.items():
                # ★ 必须复用影像的 starts：独立随机裁剪会让输入与标签空间错位
                pm, _ = crop_patch(m[None].astype(np.float32), self.patch, self.rng,
                                   starts=starts)
                patch_masks[role] = (pm[0] > 0.5).astype(np.uint8)

        if self.train and patch_masks:
            # 几何增强必须**同时**作用于影像与全部掩码角色；
            # 若只变换影像，core/peri 就会与影像错位（且不会报错）。
            roles = sorted(patch_masks)
            stack = np.stack([patch_masks[r] for r in roles]).astype(np.float32)
            patch_vol, stack = augment(patch_vol, stack, self.rng, self.aug_cfg)
            if stack is not None:
                for i, r in enumerate(roles):
                    patch_masks[r] = (stack[i] > 0.5).astype(np.uint8)
        elif self.train:
            patch_vol, _ = augment(patch_vol, None, self.rng, self.aug_cfg)

        item = {"accession": case["accession"], "image": patch_vol}
        gv = self.global_image(vol, masks)
        if gv is not None:
            item["image_global"] = gv
        item.update(self.build_label(case, patch_masks, patch_vol.shape[1:]))
        return item

    def global_image(self, vol: np.ndarray, masks: dict) -> np.ndarray | None:
        """产出**整脑视图**（全局头的输入）；返回 ``None`` 表示本次不产出。

        与 ``tasks/_common/volume.global_view`` 保持同一尺度（``size_mm=192 → out=96``），
        这样训练学到的全局头在推理侧才落在同一个输入分布上。

        中心点选择：训练且有掩码时用**病灶质心**（与算法工程一致）；
        否则交给 ``global_view`` 用前景包围盒中心（推理时只有这一种可能，
        因此训练后期也可按 ``center_jitter`` 混入该口径以增强鲁棒性）。
        """
        if not self.gv.get("enabled", True):
            return None
        from shared.volume import global_view

        center = lesion_center(masks) if (self.train and masks) else None
        if self.train and center is not None and self.gv.get("center_jitter", 0.0):
            j = float(self.gv["center_jitter"])
            center = center + self.rng.uniform(-j, j, size=3)     # 模拟中心偏差
        return global_view(vol, size_mm=float(self.gv.get("size_mm", 192)),
                           out=int(self.gv.get("out", 96)),
                           spacing=float(self.common_spacing[0]), center=center)


def starts_to_center(starts, patch):
    """把 patch 的左上角偏移换算成中心坐标（保证影像与掩码同位置裁剪）。"""
    return np.asarray(starts, float) + np.asarray(patch, float) / 2.0
