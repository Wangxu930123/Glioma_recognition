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

#: 大赛检查号的形态：**32 位小写十六进制**（实测 ``0050d79429cf4d86907dc8c4a34cbf04``）。
#:
#: 这是"这一层就是大赛数据的病例目录"的**唯一判据**，也是"只读大赛数据"的执行点：
#: 平台上 ``<数据根>/<检查号>/<序列UID>/…``，检查号必然是它。任何不满足的目录
#: （别的数据集、随手一个堆着 nii 的目录）都在**读取之前**被挡掉 ——
#: 读了会污染训练与指标，报出来会把排查方向带偏，两者都不允许。
ACCESSION_RE = re.compile(r"^[0-9a-f]{32}$")


def is_official_accession(name: object) -> bool:
    """该名字是大赛检查号吗（32 位十六进制；大小写不敏感）。"""
    return bool(ACCESSION_RE.match(str(name or "").strip().lower()))


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
#: 已就"列名与取值都认不出类型表"告警过（每个进程一次）
_WARNED_SERIES_TYPE_DEP = False

#: 类型表的列名候选（顺序即优先级；用**包含**匹配，故短词靠后）。
#:
#: 「序列描述」列三种写法都认：格式说明写 ``DetailDescription``、数据集实测拼写是
#: ``SeriesDescription``（与 DICOM 标签 (0008,103E) 同名）、另有分层写法
#: ``Study->IMAGE->序列描述``。这是官方列名里原先唯一没被认的一条，
#: 漏掉即"表找到了、却整表 0 条"。末位 ``serisdescription`` 是历史笔误的兜底。
#: 与算法工程 ``glioma_track4/src/data/labels.py`` 的同名表**逐字一致**：
#: 两条路线对同一份数据必须给出同样的模态，别名表分叉就是静默的模态错位。
_SERIES_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "acc": ("accessionnumber", "accession", "检查号", "检查编号", "病例号"),
    "uid": ("seriesinstanceuid", "seriesuid", "序列号", "序列uid"),
    "typ": ("seriestype", "type", "序列类型", "模态", "序列描述",
            "detaildescription", "seriesdescription", "serisdescription"),
}

#: 表头行扫描行数。**不要假设表头在第几行**：实测数据里第 1~3 行都可能是
#: "索引信息"（标题 / 字段说明 / 空行），写死 ``rows[0]`` 会在换一版排版时
#: 整表读成 0 条 —— 表现是"文件明明找到了，序列类型却全是空"。
_HEADER_SCAN_ROWS = 20

#: 模态取值的长度上限：``T1CE（增强）`` 也就 8 个字符，检查号 / UID 这类长串必不是模态。
_MODALITY_VALUE_MAXLEN = 16

#: "像模态取值"的额外词：它们本身不是模态（``其他`` 是**权威排除**，
#: 表示该病例确实没有目标序列），但出现在类型列里说明"这一列就是类型列"。
_OTHER_VALUE_TOKENS = frozenset({
    "其他", "其它", "无", "没有", "other", "none", "na", "n/a", "正常", "平扫",
})


def _looks_like_modality_value(value) -> bool:
    """该单元格"看着像模态取值"吗（专供列名认不出时的取值嗅探）。

    判据与模态匹配**同源**（:func:`shared.selector.guess_modality`）：
    嗅探认定"这列是类型列"的取值，后面也得能被模态匹配认出来。
    两套口径不一致的话，会出现"列挑对了、模态仍全空"这种最难查的情形。
    """
    text = re.sub(r"[\s\-_/]+", "", str(value if value is not None else "")).casefold()
    if not text or len(text) > _MODALITY_VALUE_MAXLEN:
        return False
    return guess_modality(text) is not None or text in _OTHER_VALUE_TOKENS


def _sniff_series_type_columns(rows: list, max_scan: int = 300
                               ) -> tuple[int, int, int] | None:
    """**不看列名**，按取值找出 ``(检查号列, 序列号列, 类型列)``；认不出返回 ``None``。

    为什么需要它：列名是唯一会被"改版"的东西 —— 前两列写成 ``编号/影像号``、
    加了索引列、或干脆是 ``A/B/C``，按列名匹配就一条也读不到，而**取值**不会变
    （检查号、DICOM UID、5 类模态取值）。

    判据（全在取值上）：

    * **类型列**：该列非空取值里"像模态取值"的比例最高且 ≥ 0.5；
    * **序列号列**：剩下两列里取值含 ``.``（DICOM UID）比例更高 / 平均更长的那个；
    * **检查号列**：另一个。

    找不到（整表只有两列、类型列是自由文本…）返回 ``None``，绝不硬凑 ——
    凑错会把整表挂到错误的键上，比读不到更难查。
    """
    body = [r for r in rows[:max_scan] if any(str(c).strip() for c in r if c is not None)]
    if len(body) < 3:
        return None
    width = max(len(r) for r in body)
    if width < 3:
        return None
    cols = [[str(r[c]).strip() for r in body if c < len(r) and str(r[c]).strip()]
            for c in range(width)]
    ratio = [(sum(1 for v in vals if _looks_like_modality_value(v)) / len(vals)
              if vals else 0.0) for vals in cols]
    typ_col = max(range(width), key=lambda c: (ratio[c], len(cols[c])))
    if ratio[typ_col] < 0.5:
        return None
    rest = [c for c in range(width) if c != typ_col and cols[c]]
    if len(rest) < 2:
        return None

    def dotted(c: int) -> float:
        return sum(1 for v in cols[c] if "." in v) / len(cols[c])

    def mean_len(c: int) -> float:
        return sum(len(v) for v in cols[c]) / len(cols[c])

    rest.sort(key=lambda c: (dotted(c), mean_len(c)), reverse=True)
    uid_col, acc_col = rest[0], rest[1]
    return acc_col, uid_col, typ_col


def _norm_key(value) -> str:
    """归一化用于查表的键（去空白 + 大小写无关），与提交工程同一规则。"""
    return re.sub(r"\s+", "", str(value if value is not None else "")).casefold()


def read_series_types(root: Path) -> dict[tuple[str, str], str]:
    """读数据信息表 ``SeriesType.xlsx`` → ``(检查号, 序列号) → 序列类型``。

    ⚠️ **数据里这是模态的唯一来源**。序列目录名是 DICOM UID
    （``2.25.135...``），任何"按名字猜模态"的关键词都命中不了，
    于是出现"扫出几千例、却一例都没有可用序列"——但病例计数看起来完全正常，
    很容易被误判成数据损坏或路径写错。

    表就在**数据集里**、与病例目录**同层**：训练集 ``<阶段>/annotation/``、
    验证集 ``<阶段>/original/``（实测 ``verification/original/`` 里影像、
    ``SeriesType.xlsx``、标注表是放在一起的，所以数据根填 ``…/verification``
    也要能找到）。表头实测 ``AccessionNumber`` / ``SeriesUid`` / ``SeriesType``；
    取值 **5 类**：``T1`` / ``T1CE（增强）`` / ``T2-Flair`` / ``T2WI`` / ``其他``。
    训练集、验证集都有，**评测集随测试数据一起下发**。

    读取上**不假设排版**：表头行由"能否凑齐三列名"在**前 20 行里扫**出来
    （第 1~3 行都可能是索引信息，写死 ``rows[0]`` 会在换一版排版时整表读成 0 条）；
    列名一条都不命中时还会按**取值**嗅探三列（见 :func:`_sniff_series_type_columns`）。

    定位见 :func:`shared.official_labels.label_search_dirs`：显式 ``labels_dir`` /
    ``GLIOMA_LABELS_DIR`` → 数据根/父/祖父 → 这些目录下像标注容器的一级子目录
    （``annotation`` / ``original`` / ``标注结果`` …）。
    **不再隐式搜 ``<工程>/labels`` 与 ``$WORKSPACE``**（会串表）。
    只认数据根那一层会翻车（数据根常指到 ``training/`` 或某个病例目录）。
    表里检查号列与磁盘目录名对不上**不再是问题**：查表走两级口径
    （精确键 → SeriesUid 单键回退，见 :func:`shared.official_labels.lookup_series_type`）。

    与提交工程 ``data/metadata.py: read_series_types`` 保持同一语义：
    表头别名容错、缺文件返回空表（不是错误）、同一键冲突取值**直接失败**。

    ⚠️ **工作区那份 ``工作区兼容表`` 不再参与**（这里曾把它当"只补缺、
    不覆盖"的兜底）。两个原因：①它取值粗（``SeriesLabel`` 只有 ``T1CE``/``T2``/
    ``FLAIR``），混进来会把数据集的 ``T2WI``/``T2-Flair`` **静默压成 ``T2``** ——
    表现是"模态看着都认出来了、通道里却是错的对比度"，比报错难查得多；
    ②它属于**另一个目标**的产物，用它会让"表在哪"这件事有两套答案。
    读不到就是读不到：返回空表 → 上层响亮地报 `series_type_rows = 0`
    （带 :func:`describe_modality_sources` 自检），而不是悄悄换个来源顶上。

    找不到 openpyxl 时给出**一次性显式告警**：静默返回空表会让人去改
    真正没错的地方（数据布局），而问题其实只是缺个依赖。同理，文件找到了、
    却既没认出列名也没嗅探出取值时，也会**显式告警**而不是静默返回空表。
    """
    global _WARNED_NO_OPENPYXL, _WARNED_SERIES_TYPE_DEP

    from shared.official_labels import (find_named_table,
                                        find_series_type_table_in_data)

    out: dict[tuple[str, str], str] = {}

    # 位置与影像同层：`<阶段>/annotation/SeriesType.xlsx`（训练/验证集已下发，
    # 评测集在正式测试时随测试数据一起下发）。
    # ★ **数据优先**：先在数据目录里直接找（root 本身 / annotation/ / original/
    #   父目录），找不到才退回**显式**指定的 $GLIOMA_LABELS_DIR。
    #   **不再隐式搜 <工程>/labels 与 $WORKSPACE**：那会**串表** —— 工作区 labels/ 里
    #   若残留另一份数据的 SeriesType.xlsx（如训练集的拷贝），它会先被命中 ——
    #   拿训练集的检查号/序列号去查验证集，一条都对不上
    #   （见 shared.official_labels.find_series_type_table_in_data）。
    hit = (find_series_type_table_in_data(root)
           or find_named_table("SeriesType.xlsx", root))
    path = Path(hit) if hit else None
    if path is None:
        return out
    try:
        from openpyxl import load_workbook
    except ImportError:                                           # pragma: no cover
        if not _WARNED_NO_OPENPYXL:
            _WARNED_NO_OPENPYXL = True
            print(f"[data][告警] 发现 {path} 但未安装 openpyxl，序列类型读不到 → "
                  f"UID 命名的序列会全部认不出模态。请先 pip install openpyxl",
                  flush=True)
        return out

    aliases = _SERIES_TYPE_ALIASES
    workbook = load_workbook(path, read_only=True, data_only=True)
    seen_here: dict[tuple[str, str], str] = {}                     # 仅用于本文件内冲突检测
    try:
        for sheet in workbook.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            # ① 按**列名**扫表头行：哪一行能凑齐三列就用哪一行，不假设第几行。
            #    数据里第 1~3 行都可能是索引信息，写死 rows[0] 会整表读成 0 条。
            idx: dict[str, int] = {}
            data_start = 0
            for i, row in enumerate(rows[:_HEADER_SCAN_ROWS]):
                if not any(str(c).strip() for c in row if c is not None):
                    continue
                header = [_norm_key(c) for c in row]
                found: dict[str, int] = {}
                for want, keys in aliases.items():
                    for j, h in enumerate(header):
                        if any(k in h for k in keys):
                            found[want] = j
                            break
                if set(found) == {"acc", "uid", "typ"}:
                    idx, data_start = found, i + 1
                    break
            sniffed = False
            if not idx:
                # ② 列名一条都没命中（改版 / 加了索引列 / 只有英文缩写）
                #    → 按**取值**嗅探三列（检查号、DICOM UID、模态取值本身就有形态）。
                sniff = _sniff_series_type_columns(rows)
                if sniff is None:
                    continue                                      # 该 sheet 不是映射表
                acc_col, uid_col, typ_col = sniff
                idx = {"acc": acc_col, "uid": uid_col, "typ": typ_col}
                data_start, sniffed = 0, True
                print(f"[data][告警] {path.name} 的工作表 {sheet.title!r} 列名未识别 → "
                      f"已按取值定位：检查号=第 {acc_col + 1} 列、序列号=第 {uid_col + 1} 列、"
                      f"类型=第 {typ_col + 1} 列（读到 {len(rows)} 行）。"
                      f"若取值明显不对，把表的前几行贴出来。", flush=True)
            for row in rows[data_start:]:
                try:
                    acc, uid, typ = row[idx["acc"]], row[idx["uid"]], row[idx["typ"]]
                except IndexError:
                    continue
                if acc in (None, "") or uid in (None, "") or typ in (None, ""):
                    continue
                if sniffed and not _looks_like_modality_value(typ):
                    continue                                      # 嗅探模式下靠取值滤掉表头/说明行
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
        print(f"[data] 已读数据信息表 {path.name}：{len(seen_here)} 条（{path}）", flush=True)
    elif not _WARNED_SERIES_TYPE_DEP:
        # 文件在、openpyxl 也在，却一条都没读到：只可能是"排版对不上"。
        # 静默返回空表会让人去怀疑数据损坏/路径写错，所以这里明确指向排版，
        # 并说明该看什么（表的前几行），而不是丢一句"序列类型读不到"。
        _WARNED_SERIES_TYPE_DEP = True
        print(f"[data][告警] {path} 里既没找到 AccessionNumber / SeriesUid / SeriesType "
              f"三列（已扫前 {_HEADER_SCAN_ROWS} 行）、也没能按取值嗅探出它们 → "
              f"序列类型读不到，UID 命名的序列会全部认不出模态。"
              f"请把该表前几行原样贴出来。", flush=True)
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


#: 官方**标注结果表**的文件名（含 `ROI级别` sheet 的「序列描述」列）。
#: 它是模态的**第二条来源**：类型表没到手 / 表里没有这一批 UID 时兜底。
_ANNOTATION_RESULT_TABLES = ("脑胶质瘤标注结果-训练集.xlsx", "脑胶质瘤标注结果-验证集.xlsx")

#: 「序列描述」列的候选列名。与 :data:`_SERIES_TYPE_ALIASES` 的 ``typ`` 同源 ——
#: 那一列**本身就是模态取值**（``T1`` / ``T2-Flair`` / ``T1CE（增强）`` / ``其他``）。
#: 官方口径叫 ``DetailDescription``，数据集实测拼写 ``SeriesDescription``
#: （ROI 级别 sheet），分层写法末段 ``序列描述``。
_SERIES_DESC_KWS: tuple[str, ...] = ("seriesdescription", "serisdescription",
                                     "detaildescription", "seriestype",
                                     "序列描述", "序列类型")

#: DICOM UID 只含数字与点。用它把"UID 列"与"描述列"区分开：
#: :func:`shared.official_labels._find_col` 的"包含"轮里，``series`` 会命中
#: ``SeriesDescription`` **自己**，于是两列落成同一列、取值互相顶替。
_UID_LIKE_RE = re.compile(r"^\d[\d.]*$")


def read_series_desc_index(root) -> dict[str, str]:
    """标注结果表 → ``{序列UID(归一化): 序列描述}``（**模态旁证**）。

    为什么需要它（与算法工程 ``glioma_track4/src/data/labels.desc_index_from_records``
    **同一口径**）：模态的权威来源是数据自带的 ``SeriesType.xlsx``，但它**可能没到手，
    或表里恰好没有这一批 UID**。此时唯一还能**批量**判模态的正规线索，就是标注结果表
    ``ROI级别`` sheet 的「序列描述」列 —— 行键是 ``序列Uid``，与磁盘上的序列目录名 /
    文件名主干**同源**，能直接对上。

    两条路线（本工程与 ``glioma_track4``）必须对同一份数据给出**同样的模态**：
    只有一边认得出、另一边认不出，就是"训练与推理喂的模态不一致"这类静默错位。

    读法与类型表一致：不假设表头在第几行（多张表拼在一起，每张各自一段表头），
    列名认不出就跳过；取值必须**像模态**才收（否则散文列会被当成描述列）。
    找不到表 / 认不出列都返回空表（由上层报数），不抛异常。
    """
    from shared.official_labels import _find_col, _rows, find_named_table

    out: dict[str, str] = {}
    path = None
    for name in _ANNOTATION_RESULT_TABLES:
        path = find_named_table(name, root)
        if path:
            break
    if not path:
        return out
    uid_col: int | None = None
    desc_col: int | None = None
    for row in _rows(path):
        cells = [str(c).strip() for c in row]
        if not any(cells):
            continue
        # 每一段表头都重新定位两列（`_rows` 把多张工作表拼成了一个列表）
        _u = _find_col(cells, _SERIES_TYPE_ALIASES["uid"])
        _d = _find_col(cells, _SERIES_DESC_KWS)
        if _u is not None and _d is not None and _u != _d:
            uid_col, desc_col = _u, _d
            continue
        if uid_col is None or desc_col is None:
            continue
        if max(uid_col, desc_col) >= len(cells):
            continue
        uid, desc = cells[uid_col], cells[desc_col]
        # 两列都要过形态校验：UID 是数字+点；描述要像模态（含"其他"这类权威排除值）
        if _UID_LIKE_RE.match(uid) and _looks_like_modality_value(desc):
            out.setdefault(_norm_key(uid), desc)
    return out


def _series_desc(path: Path, accession: str, uid: str, series_types: dict,
                 uid_index: dict | None = None,
                 desc_index: dict | None = None) -> str:
    """序列类型的**取用优先级**：类型表 → **标注表「序列描述」旁证** → sidecar → 目录名。

    返回的文本会作为 ``Series`` 的 ``modality`` 交给模态关键词匹配，
    因此它可以是 ``T1CE`` / ``FLAIR`` / ``T1增强`` 这类**任意自然描述**。

    查表时**目录名与文件名都试**：官方数据的组织方式不止一种 ——
    ``<检查号>/<序列号>/<序列号>.nii.gz``（序列号在目录名上）
    与 ``<检查号>/<序列号>.nii.gz``（序列号在文件名上）都可能出现，
    只试目录名会让后者整批认不出模态。

    查表走两级（:func:`shared.official_labels.lookup_series_type`）：先
    ``(检查号, 序列号)`` 精确键，查不到再按 **SeriesUid 单键回退** ——
    表里的检查号列与磁盘目录名口径不一致时（匿名化/哈希），精确键会整批落空。

    ``desc_index`` 是 :func:`read_series_desc_index` 的产出（第二条来源，可选）：
    类型表**整份缺失**时它仍按 UID 直接命中，是"类型表没到手"唯一的批量退路。
    """
    from shared.official_labels import lookup_series_type

    stem = path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem
    value = lookup_series_type(series_types, accession, (uid, stem), uid_index)
    if value:
        return str(value)
    if desc_index:
        for cand in (uid, stem):
            hit = desc_index.get(_norm_key(cand))
            if hit:
                return str(hit)
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
#:
#: ``original`` 是**验证集**的实测布局：``verification/original/<检查号>/<序列>/``，
#: 影像、``SeriesType.xlsx`` 与标注表都在 ``verification/original/``。漏了它，
#: 数据根填 ``…/verification`` 时 ``original`` 会被当成检查号 —— 表现是
#: "病例数正常、却报无任何可用序列"，且表也找不到。
_DESCEND_DIRS = frozenset({"annotation", "original"}) | _PLATFORM_PHASES
#: 顶层非病例目录：本层出现其中任何一个，说明"还没到病例层"
_NON_CASE_DIRS = (frozenset({"annotation", "original", "cache", "runs", "folds",
                             "labels", "logs", "checkpoints", "weights"})
                  | frozenset(SPECIAL_SOURCE_DIRS))


def resolve_case_root(root):
    """把"填高了一层"的数据根下钻到真正含病例目录的那一层。

    平台实测（2026-09-24）——**训练集与验证集的中间层名字不同**：

    ```text
    /2026aicompetition/datasets/training/annotation/<检查号>/…      ← 容器叫 annotation
    /2026aicompetition/datasets/verification/original/<检查号>/…    ← 容器叫 original
    ```

    填高一层不报错、只静默扫到 0 例（训练照常启动、损失照常不动）。
    规则与算法工程 ``src/data/probe.py::resolve_case_root`` 一致：

    1. 唯一子目录是"容器"（已知容器名，**或其内直接放着 SeriesType.xlsx**）
       → 下钻并告警 —— 后半条**不依赖容器名**，评测集 ``evaluation_*``
       的容器叫什么都覆盖；
    2. 本层存在"非白名单"的子目录 → 病例层（本地模拟集、无表的布局）；
    3. 已知容器名的唯一候选 → 下钻。

    多阶段父目录交给 :func:`assert_case_root` 报错，不下钻、不猜。
    """
    root = Path(root)
    if not root.is_dir():
        return root
    children = sorted(p for p in root.iterdir() if p.is_dir())
    if not children:
        return root

    def _is_container(p: Path) -> bool:
        if p.name.casefold() in _DESCEND_DIRS:
            return True
        # 名字无关的判定：容器里直接放着数据信息表（与检查号目录同层）
        return (p / "SeriesType.xlsx").is_file()

    if len(children) == 1 and _is_container(children[0]):     # ① 唯一子目录是容器
        print(f"[data][告警] 数据根 {root} 下没有病例目录，已自动下钻到 "
              f"{children[0].name}/（若不对请用 --data / GLIOMA_DATASET_ROOT 指定）",
              flush=True)
        return children[0]
    if any(p.name.lower() not in _NON_CASE_DIRS for p in children):
        return root                                       # ② 本层已经有病例目录
    cands = [p for p in children if p.name.casefold() in _DESCEND_DIRS]
    if len(cands) == 1:                                   # ③ 已知容器名的唯一候选
        print(f"[data][告警] 数据根 {root} 下没有病例目录，已自动下钻到 "
              f"{cands[0].name}/（若不对请用 --data / GLIOMA_DATASET_ROOT 指定）",
              flush=True)
        return cands[0]
    return root


def _official_context(root: Path) -> dict:
    """一次性读取官方标注表（``1_/2_/4_/5_`` 四张；缺失的键为默认空值）。

    这是研发侧**唯一权威**的标签来源：掩膜、结构化字段、异常标记
    全都来自它，而不是 ``label.json``、目录名关键词或中文列名——
    官方数据里那些都不存在，于是 special 标签恒为 0、字段全空，
    训练照常跑完却什么都没学到（最难发现的一类失效）。

    模态**不在这里**：它来自数据集自带的 ``SeriesType.xlsx``
    （见 :func:`read_series_types`）。
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


def discover_cases(dataset_root: Path, limit: int | None = None,
                   diagnose: bool = True) -> list[dict]:
    """扫描 ``<root>/<AccessionNumber>/<SeriesUid>/*.nii[.gz]``。

    返回的每个 case 除 ``accession``/``dir``/``series`` 外，还可能带：

    - ``series[].desc``：序列类型描述（数据信息 `SeriesType.xlsx` → sidecar → 目录名）
    - ``masks``：按角色分组的掩膜（官方 `4_masklabel.xlsx` 的 Maskname 或名称启发）
    - ``labels``：结构化字段（官方 `5_characteristics.xlsx`，已是规范字段名）
    - ``special``：``{fake, stitched, duplicate}``（官方 `1_abnormal.xlsx`）
    - ``source``：``true`` / ``fake`` / ``compositing`` / ``duplicate``（影像所在子目录）

    只做轻量发现（不读体素），真正的读取延迟到 ``load_case``。

    ``diagnose=False``：**探测模式** —— 调用方在对**多个候选根**轮流试（如
    ``glioma_track4/scripts/29_locate_dataset_root.py``），此时"模态认不出"的诊断
    折叠成一行，由调用方汇总；避免把无关根的多行告警混进结果里，看着像在说官方数据集坏了。
    无论开关如何，**每次调用都会打印"本批是哪个根 + 表读到几条"**，多根循环时不再串场。
    """
    root = resolve_case_root(Path(dataset_root))   # 填高一层（如 .../training）时自动下钻
    assert_case_root(root)
    # ---- 大赛数据闸门：只认 32 位十六进制检查号（其余目录**不读、不打印**）----
    # 平台契约：``<数据根>/<32 位十六进制检查号>/<序列UID>/…``。非大赛目录在这里
    # 就被挡掉 —— 既不会进训练，也不会在日志里出现（出现只会把排查方向带偏）。
    _dirs = [p for p in root.iterdir() if p.is_dir()]
    if not _dirs:
        print(f"[data] ⚠️ 数据根 {root} 下没有任何子目录 → 0 例"
              f"（影像存储可能没挂上，见 docs/DATASET_ROOT_TROUBLESHOOT.md）", flush=True)
        return []
    _case_dirs = [p for p in _dirs if is_official_accession(p.name)]
    _own_dirs = [p for p in _dirs if p.name.casefold() in
                 (frozenset(_DESCEND_DIRS) | frozenset(SPECIAL_SOURCE_DIRS)
                  | frozenset(_NON_CASE_DIRS))]
    if not _case_dirs:
        raise ValueError(
            f"数据根 {root} 不是大赛数据布局：一级子目录里没有 32 位十六进制的检查号"
            f"（如 0050d79429cf4d86907dc8c4a34cbf04）。当前 {len(_dirs)} 个子目录、"
            f"其中 {len(_own_dirs)} 个是本工程自己的目录（annotation/original/"
            f"fake/compositing/duplicate 等）。官方布局：训练集根="
            f"…/datasets/training（其下 annotation/）、验证集根="
            f"…/datasets/verification（其下 original/），病例目录名就是检查号本身。")
    from shared.official_labels import (SERIES_TYPE_TABLE, build_uid_index,
                                        find_named_table,
                                        find_series_type_table_in_data)

    series_types = read_series_types(root)
    # 本批**真正用到**的那张表（与 `read_series_types` 的查找口径完全一致）。
    # 报告它、而不是"重新搜一遍文件系统"，否则会出现"表读到了 N 条 / 自检说未找到"
    # 的自相矛盾（实测踩过：多根循环时把另一个根搜空，被当成本批的结论）。
    _type_path = (find_series_type_table_in_data(str(root))
                  or find_named_table(SERIES_TYPE_TABLE, str(root)))
    # 表里的检查号列与磁盘目录名对不上时，靠它按 SeriesUid 单键回退（见 _series_desc）
    uid_index = build_uid_index(series_types)
    # 模态的**第二条来源**（与 `glioma_track4` 同口径）：标注结果表 `ROI级别` 的
    # 「序列描述」。类型表整份缺失时，它是唯一还能**批量**判模态的正规线索 ——
    # 缺了它就只能靠目录名（UID 命名下必落空）→ 全零通道照训（白跑且不报错）。
    desc_index = read_series_desc_index(root)
    # ★ 每次调用都**报出是哪个根**：`discover_cases` 会被探测类脚本对多个候选根轮流调用，
    #   不打根就无法判断后面每段诊断属于谁（这正是"6340 条 / 未找到"并存的成因）。
    print(f"[data] ── 数据根 {root}｜数据信息表 {_type_path or '未找到'}"
          f"（{len(series_types)} 条）｜序列描述旁证 {len(desc_index)} 条", flush=True)
    if desc_index:
        print(f"[data] 已从标注表读到「序列描述」（模态旁证）：{len(desc_index)} 条序列 UID",
              flush=True)
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
            desc = _series_desc(f, acc_dir.name, uid, series_types, uid_index, desc_index)
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
    # 迭代 `_case_dirs`（已在闸门里筛成 32 位十六进制）：容器目录（annotation/
    # original）与特殊影像目录（fake/…）天然不在其中，不必再按名字排除。
    for acc_dir in _case_dirs:
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
        for acc_dir in sorted(p for p in src_dir.iterdir()
                              if p.is_dir() and is_official_accession(p.name)):
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

    # 前置预警：一条序列都认不出模态时的诊断。
    #
    # ★★ 两条**必须**遵守的写法（都踩过坑）：
    #   1. **必须写明数据根**：`discover_cases` 常被"探测类"脚本对**多个候选根**轮流调用
    #      （`scripts/29_locate_dataset_root.py` 一次试多个根），不写根就会出现
    #      "上一行说读到 N 条、这一行说未找到"的自相矛盾 —— 那其实是**两个不同的根**。
    #   2. **只能用手里的事实**：早先这里会**重新搜一遍文件系统**（调 official_labels 里
    #      的"模态来源自检"），它只回答"表在不在"，既可能与 `series_types`（本批真正
    #      用到的表）不一致，也看不出"表在、键对不上"这种更常见的情形。现在直接报告
    #      **本批读到的表**（路径 + 条数）与**键命中数**，两种成因分开说。
    if cases and not any(
        guess_modality(s.get("desc") or s.get("uid") or "")
        for c in cases[:50] for s in c.get("series") or []
    ):
        _disk_accs = [c["accession"] for c in cases[:2]]
        _disk_uids = [s.get("uid") for c in cases[:2]
                      for s in (c.get("series") or [])][:3]
        _acc_hit = sum(1 for c in cases[:50]
                       if _norm_key(c["accession"]) in
                       {k[0] for k in series_types if isinstance(k, tuple)})
        _uid_hit = sum(1 for c in cases[:50] for s in (c.get("series") or [])
                       if _norm_key(s.get("uid") or "") in
                       {k[1] for k in series_types if isinstance(k, tuple)})
        if not diagnose:
            # 探测模式（一次试很多候选根）：折叠成一行，别刷屏、别看起来像结论。
            print(f"[data] · 数据根 {root}：前 50 例没有序列能识别出模态"
                  f"（表 {'有' if series_types else '无'} / {len(series_types)} 条，"
                  f"检查号命中 {_acc_hit} 例）—— 探测模式下仅记录，见调用方的汇总",
                  flush=True)
        else:
            if series_types:
                _why = (f"       本批数据信息表：{_type_path}（{len(series_types)} 条）"
                        f"—— **表读到了，但键一条都命中不上**："
                        f"前 50 例里检查号命中 {_acc_hit} 例 / 序列号命中 {_uid_hit} 路。\n"
                        "       两种成因：① **串表**（这份表属于另一批数据，"
                        "如拿训练集的表查验证集）；② **键口径不一致**"
                        "（表里的检查号/序列号列与磁盘目录名对不上）。\n")
            else:
                _why = ("       本批**没有**读到数据信息表（数据根/父/祖父 + 标注容器子目录里都没有）。\n"
                        "       表若确实在别处：export GLIOMA_LABELS_DIR=<含表目录>。\n")
            print(f"[data] ⚠️ 数据根 {root}：前 50 例没有任何序列能识别出模态"
                  f"（序列目录名/文件名不含模态关键词）。\n"
                  + _why +
                  f"       磁盘样例：检查号 {_disk_accs}、序列目录 {_disk_uids}\n"
                  f"       ② 级（标注表「序列描述」旁证）读到 {len(desc_index)} 条 UID："
                  "0 条 = 标注结果表没找到 / 没有「序列描述」列。\n"
                  "       ⚠️ 全放开口径下，继续训练**不会再报错** —— 它会把 4 个通道全置零照训"
                  "（借该例影像几何）。所以这不再是「会不会崩」，而是**训练会白跑**："
                  "输入是常数，模型学不到东西，指标只会悄悄变差（比报错难查得多）。\n"
                  "       处理（按序）：① 确认数据根没填偏 —— 表与病例目录**同层**"
                  "（training→annotation、verification→original）；"
                  "② 表在非常规位置 → export GLIOMA_LABELS_DIR=<它所在目录>；"
                  "③ 类型表没有可取的值 → 用标注结果表的「序列描述」（本工程已自动接）；"
                  "④ 以上都没有 → 先训体素判别模型："
                  "python glioma_track4/scripts/31_train_modality_model.py --root <数据根>",
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
