"""官方标注表读取（依据天坛医院参考实现 ``AIRecongition/``）。

官方把标注放在**工程目录**的 ``labels/`` 下（``paths.labels_dir: ./labels``），
是 4 个独立 xlsx（外加**数据集自带**的数据信息表 ``SeriesType.xlsx``）：

| 文件 | 关键列 | 用途 |
|---|---|---|
| ``1_abnormal.xlsx`` | AccessionNumber, SeriesUid, **Label** ∈ {true,fake,compositing,duplicate} | 目标一（真实性）/ 目标二-A（拼接）的正样本，**逐序列** |
| ``2_duplicate.xlsx`` | src_img, desc_img | 目标二-B（重复影像）的正对 |
| ``SeriesType.xlsx`` | AccessionNumber, SeriesUid, **SeriesType** ∈ {T1, T1CE（增强）, T2-Flair, T2WI, 其他} | **赛道四数据集自带的数据信息**：与病例目录同层（``<阶段>/annotation/``），训练/验证集都有、评测集随测试数据下发；取值里的 ``其他`` 是权威排除（不是模态）。**模态的唯一来源** |
| ``4_masklabel.xlsx`` | AccessionNumber, SeriesUid, **Maskname** | 掩膜文件名（任意名，关键词认不出） |
| ``5_characteristics.xlsx`` | AccessionNumber + 14 个英文列 | 目标三/四的结构化字段金标准 |

> ``工作区兼容表``（团队工作区那份）**不在本模块读取范围内**：它不是赛道四
> 数据集的内容，且取值更粗（只有 ``T1CE``/``T2``/``FLAIR``）—— 混用会把数据集的
> ``T2WI``/``T2-Flair`` **静默压成 ``T2``**，表现是"模态看着都认出来了、通道里却是
> 错的对比度"，比直接报错难查得多。它曾作为兜底出现在 ``read_series_labels`` 里，
> 现已连函数一起删除；照旧按候选目录搜它的人只会读到"找不到"。

在此之前，研发侧是从 ``label.json``、目录名关键词、中文列名去猜的 ——
官方数据里这些都不存在，于是 special 标签恒为 0、字段全空，
训练照常跑完却什么都没学到。本模块把官方口径固定下来。
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

#: 官方标注文件名 → 用途（键名与官方一致，便于对照）
#:
#: **不含** ``工作区兼容表``：模态来自数据集自带的 ``SeriesType.xlsx``
#: （见模块 docstring）。留"series"这个键会让调用方以为还有第二个来源可选。
OFFICIAL_LABEL_FILES = {
    "abnormal": "1_abnormal.xlsx",
    "duplicate": "2_duplicate.xlsx",
    "mask": "4_masklabel.xlsx",
    "characteristics": "5_characteristics.xlsx",
}

#: 环境变量（对应官方 config 的 ``paths.labels_dir``）
LABELS_DIR_ENV = "GLIOMA_LABELS_DIR"

#: 序列类型表（数据集自带的数据信息）的文件名。
SERIES_TYPE_TABLE = "SeriesType.xlsx"

#: 数据目录里序列类型表可能待的位置：**与检查号目录同层**（平台契约），
#: 容器名训练集是 ``annotation``、验证集是 ``original``。
_DATA_TABLE_SUBDIRS = ("", "annotation", "original")


def find_series_type_table_in_data(root: str | os.PathLike | None) -> str | None:
    """**只在数据目录里**找序列类型表 → 路径或 ``None``。

    为什么需要它（防**跨数据集串表**）：:func:`find_named_table` 的候选顺序是
    ``显式 labels_dir/$GLIOMA_LABELS_DIR → 数据根/父/祖父``
    （早先还会搜 ``<工程>/labels`` 与 ``$WORKSPACE``，已移除）。
    若别处残留了一份**另一个数据集**的 ``SeriesType.xlsx``
    （比如把训练集的表拷过去过），它会**先于**当前数据自己的表被命中 ——
    表现正是"表读到了几千条、却一条都查不到"：拿训练集的检查号/序列号去查
    验证集的数据，而且日志里表的路径指向 ``labels/`` 而不是数据目录。

    所以模态表**数据优先**：先查 ``root`` 本身、``root/annotation``、
    ``root/original``、``root 的父目录``（都不在时才退回通用搜索）。
    其余四张表（``1_abnormal`` 等）仍走通用搜索 —— 它们本来就在工作区。
    """
    if root is None:
        return None
    base = Path(str(root)).expanduser()
    try:
        base = base.resolve()
    except OSError:
        base = base.absolute()
    for sub in _DATA_TABLE_SUBDIRS:
        folder = base / sub if sub else base
        candidate = folder / SERIES_TYPE_TABLE
        if candidate.is_file():
            return str(candidate)
    return str(base.parent / SERIES_TYPE_TABLE) \
        if (base.parent / SERIES_TYPE_TABLE).is_file() else None

#: 官方 ``5_characteristics.xlsx`` 列名 → 我们的规范字段名
OFFICIAL_FIELD_COLUMNS = {
    "Glioma": "TumorProbability",            # No / Yes
    "WHO_grade": "WHO_Grade",                # 1/2/3/4
    "Enhancement": "Enhancement",            # false / true
    "EnhancementPattern": "EnhancementPattern",
    "Necrosis": "Necrosis",
    "CysticChange": "CysticChange",
    "Hemorrhage": "Hemorrhage",
    "Calcification": "Calcification",
    "Margin": "Margin",                      # Unclear / Clear
    "Lobulation": "Lobulation",
    "Morphology": "Morphology",              # Regular / Irregular
    "Signal_T2WI": "Signal_T2WI",            # Low / Iso / High
    "Signal_FLAIR": "Signal_FLAIR",
    "Location": "Location",                  # 多标签，`|` 分隔
}

#: 官方列里属于**二分类**的（存 0/1，与训练目标一致）
OFFICIAL_BINARY_COLUMNS = {"Glioma", "Enhancement", "Necrosis", "CysticChange",
                           "Hemorrhage", "Calcification", "Margin", "Lobulation"}


# --------------------------------------------------------------------------- #
# 表定位
# --------------------------------------------------------------------------- #
#: **表的搜索范围 = 数据根（+其父/祖父 + 像标注容器的子目录）。**
#:
#: 这里曾会去 ``<工程>/labels`` 与 ``$WORKSPACE`` 下 3 层翻 ``labels/``。
#: **已移除**：①隐式翻别处会**串表**（工作区残留另一份数据的表 → 先于当前数据被命中，
#: "表读到几千条却一条都查不中"）；②本赛道数据集自带全部所需（模态 `SeriesType.xlsx`、
#: 掩膜 `_mask` 文件、字段 `脑胶质瘤标注结果-*.xlsx`、特殊/重复影像目录），
#: 那 5 张工作区表**不是本赛道数据集的内容**。
#: 仍需指向别处时用 :data:`LABELS_DIR_ENV` **显式**指定（不做任何隐式搜索）。


def find_official_labels(root: str | os.PathLike | None = None,
                         labels_dir: str | os.PathLike | None = None) -> dict[str, str]:
    """定位官方标注表（``1_/2_/4_/5_`` 四张）→ ``{用途: 路径}``（找不到的键不出现）。

    搜索顺序见 :func:`label_search_dirs`（逐个候选目录试，命中即止）。
    ``root=None`` 时只搜 1~4（用于报错时做"表到底在不在"的自检）。
    """
    found: dict[str, str] = {}
    for kind, name in OFFICIAL_LABEL_FILES.items():
        hit = find_named_table(name, root, labels_dir)
        if hit:
            found[kind] = hit
    return found


#: 候选目录里**值得下钻一层**的子目录名（大小写无关）：标注/结果类容器。
#:
#: 为什么不无脑扫全部子目录：平台数据根下有 **3000+ 病例目录**（32 位哈希），
#: 逐个 stat 既慢，又会在备份/缓存目录里撞到同名旧表。而"标注放在哪个容器目录"
#: 是个有限集合：平台那份在 ``training/annotation/``（英文）或下载后的
#: ``标注结果/``（中文），下面这些名字把两种情况都覆盖了。
#: "像标注容器"的子目录名。
#:
#: ``original`` 是**验证集**的中间层：``verification/original/`` 里既放影像也放
#: ``SeriesType.xlsx`` / ``脑胶质瘤标注结果-验证集.xlsx``。漏了它，数据根填
#: ``…/verification``（而不是 ``…/verification/original``）时表就一条都搜不到。
_LABEL_SUBDIR_NAMES = frozenset({
    "labels", "label", "annotation", "annotations", "标注结果", "标注", "结果",
    "original", "gold", "groundtruth", "ground_truth", "gt", "meta", "metadata",
    "results",
})

#: 往下钻几层。2 层是为了覆盖"数据根填高一层（``training/``）**且**
#: 表还在容器子目录里（``annotation/标注结果/``）"这种叠加情况。
_LABEL_SUBDIR_DEPTH = 2


def _subdir_candidates(base: Path) -> list[Path]:
    """``base`` 下"像标注容器"的一级子目录（读不了就返回空，不抛）。

    先按名字过滤再 ``is_dir()``：数据根下可能有 3000+ 病例目录，
    对每个条目都 stat 一次会白花几十毫秒（这里只在名字命中时才 stat）。
    """
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return []
    return [p for p in entries
            if p.name.lower() in _LABEL_SUBDIR_NAMES and p.is_dir()]


def label_search_dirs(root: str | os.PathLike | None = None,
                      labels_dir: str | os.PathLike | None = None) -> list[Path]:
    """标注表候选目录（**顺序即优先级**，命中即止）。

    1. 显式传入的 ``labels_dir``（对应官方 config 的 ``paths.labels_dir``）
    2. 环境变量 ``GLIOMA_LABELS_DIR`` —— **显式**指定，只有你亲手设了才用它
    3. 数据根自身、父目录、祖父目录 —— 官方把 ``SeriesType.xlsx`` 放在**病例目录那一层**
       （``training/annotation/``）；传进来的若是**病例目录**，父/祖父正好覆盖到它
    4. 上述目录下"像标注容器"的一级子目录（见 :data:`_LABEL_SUBDIR_NAMES`）——
       防住"表藏在数据根下一层（如 ``标注结果/``）"这种情况

    ⚠️ **不再搜 ``<工程>/labels`` 与 ``$WORKSPACE``**（会**串表**：工作区残留另一份
    数据的表 → 先于当前数据被命中，"表读到几千条却一条都查不中"）；本赛道数据集自带
    全部所需。要指向别处请用 ``GLIOMA_LABELS_DIR`` 显式说明。
    """
    cands: list[Path] = []
    if labels_dir:
        cands.append(Path(labels_dir).expanduser())
    if os.environ.get(LABELS_DIR_ENV):
        cands.append(Path(os.environ[LABELS_DIR_ENV]).expanduser())
    if root is not None:
        base = Path(str(root)).expanduser()
        try:
            base = base.resolve()
        except OSError:
            base = base.absolute()
        cands.extend([base, base.parent, base.parent.parent])

    out: list[Path] = []
    seen: set[str] = set()
    for folder in cands:
        frontier = [folder]
        for _ in range(_LABEL_SUBDIR_DEPTH + 1):
            nxt: list[Path] = []
            for cand in frontier:
                key = str(cand)
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
                nxt.extend(_subdir_candidates(cand))
            frontier = nxt
    return out


def find_named_table(filename: str, root: str | os.PathLike | None = None,
                     labels_dir: str | os.PathLike | None = None) -> str | None:
    """在候选目录里按**文件名**找一张表 → 路径（找不到返回 ``None``）。

    与 :func:`find_official_labels` 同一批候选目录，区别只在"名字由调用方给"：
    数据信息表（``SeriesType.xlsx``）与字段金标准（``脑胶质瘤标注结果-*.xlsx``）
    名字不同、位置也可能不同，用同一个搜索口径分别找。
    """
    for folder in label_search_dirs(root, labels_dir):
        candidate = folder / filename
        if candidate.is_file():
            return str(candidate)
    return None


def build_uid_index(series_types: dict | None) -> dict[str, str]:
    """``{(检查号, 序列号): 类型}`` → ``{序列号: 类型}``（UID 单键回退索引）。

    为什么需要它：类型表的**检查号列**与磁盘上的病例目录名并非总能
    对上（平台匿名化口径不同、前导零、目录名是哈希而表里是原始检查号），而
    **SeriesUid 与影像同源**，是两边唯一必然一致的键。精确键查不到时按 UID 单键回退，
    能把整批"看起来没模态"的病例救回来。

    表可能上万行：请**每轮只构建一次**，不要放进病例循环里。
    """
    out: dict[str, str] = {}
    for key, value in (series_types or {}).items():
        if not value:
            continue
        uid = key[1] if isinstance(key, (tuple, list)) and len(key) > 1 else key
        out.setdefault(_norm(uid), str(value))
    return out


def lookup_series_type(series_types: dict | None, accession: str = "",
                       uid_candidates: tuple | list = (),
                       uid_index: dict | None = None) -> str:
    """两级查表：``(检查号, 序列号)`` 精确键 → ``序列号`` 单键回退。

    ``uid_candidates`` 按可靠性降序给（如 ``(序列目录名, 文件名主干)``）。
    未传 ``uid_index`` 时本函数自行构建（单次调用用；循环里请在外面建好传进来）。
    """
    if not series_types:
        return ""
    for uid in uid_candidates:
        value = series_types.get((_norm(accession), _norm(uid)))
        if value:
            return str(value)
    index = uid_index if uid_index is not None else build_uid_index(series_types)
    for uid in uid_candidates:
        value = index.get(_norm(uid))
        if value:
            return str(value)
    return ""


def describe_modality_sources(root: str | os.PathLike | None = None) -> str:
    """一句话自检"模态来源现在什么状态"（专供报错文案，省掉一轮来回排查）。

    形如 ``数据信息表 SeriesType.xlsx=<路径或"未找到">``。
    报错带上它，用户立刻能分清是"表没接上"还是"表接到了但标注本身不含目标模态"。
    ``root`` 传**病例目录**也行：候选目录含数据根/父/祖父，正好覆盖到
    ``training/annotation/``。

    只报这一张表：模态**只有一个来源**（工作区那份 ``工作区兼容表`` 已不再被读，
    列出来只会误导排查方向）。

    查找顺序与 :func:`read_series_types` 完全一致（**数据优先**，见
    :func:`find_series_type_table_in_data`）—— 自检报的必须是真正会用到的那张表，
    否则"自检说找到了、实际读的是另一张"会把排查带偏。
    """
    path = (find_series_type_table_in_data(root)
            or find_named_table("SeriesType.xlsx", root))
    if path:
        return f"数据信息表 SeriesType.xlsx={path}"
    return ("数据信息表 SeriesType.xlsx=未找到（已搜 "
            "<数据根>/SeriesType.xlsx、annotation/、original/、父目录，"
            "以及显式指定的 $GLIOMA_LABELS_DIR；"
            "平台数据里这张表与病例目录同层）")


# --------------------------------------------------------------------------- #
# 原始行读取（不做表头假设）
# --------------------------------------------------------------------------- #
def _rows(path: str) -> list[list[str]]:
    """读一张表的**全部工作表**为原始行（csv 用单表）。"""
    if str(path).lower().endswith((".xlsx", ".xls")):
        import pandas as pd
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
        out: list[list[str]] = []
        for frame in sheets.values():
            out.extend([["" if v is None else str(v).strip() for v in row]
                        for row in frame.fillna("").values.tolist()])
        return out
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [[str(c).strip() for c in row] for row in csv.reader(f)]


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value if value is not None else "")).casefold()


#: 公开别名：查表键归一化三处共用同一套规则，各写一份迟早出现"一处去空白、一处不去"
norm_key = _norm


def _find_col(header: list[str], keywords: tuple[str, ...]) -> int | None:
    """表头里找列（返回列号）：精确 → 前缀 → 包含。"""
    lower = {str(c).strip().lower(): i for i, c in enumerate(header) if str(c).strip()}
    for kw in keywords:
        if kw in lower:
            return lower[kw]
    for kw in keywords:
        for low, idx in lower.items():
            if low.startswith(kw):
                return idx
    for kw in keywords:
        for low, idx in lower.items():
            if kw in low:
                return idx
    return None


def _detect_header(rows: list[list[str]], id_kws: tuple[str, ...]) -> int | None:
    """在前若干行里找真表头（标题行/空行很常见）。"""
    for idx, row in enumerate(rows[:20]):
        if not any(str(c).strip() for c in row):
            continue
        if _find_col([str(c) for c in row], id_kws) is not None:
            return idx
    return None


def _triples(path: str,
             id_kws: tuple[str, ...] = ("accessionnumber", "accession", "检查号"),
             uid_kws: tuple[str, ...] = ("seriesuid", "series_uid", "序列号"),
             value_kws: tuple[str, ...] = ("label",),
             ) -> dict[tuple[str, str], list[str]]:
    """读官方"检查号 + 序列号 + 取值"表 → ``{(检查号, 序列号): [取值, ...]}``。"""
    out: dict[tuple[str, str], list[str]] = {}
    rows = _rows(path)
    head = _detect_header(rows, id_kws)
    if head is None:
        return out
    header = [str(c).strip() for c in rows[head]]
    i_id, i_uid = _find_col(header, id_kws), _find_col(header, uid_kws)
    i_val = _find_col(header, value_kws)
    if i_id is None or i_uid is None or i_val is None:
        print(f"[labels][告警] {os.path.basename(path)} 缺少必需列："
              f"表头={[c for c in header if c][:8]}", flush=True)
        return out
    for row in rows[head + 1:]:
        def _cell(i: int) -> str:
            return str(row[i]).strip() if i < len(row) else ""
        acc, uid, value = _cell(i_id), _cell(i_uid), _cell(i_val)
        if not acc or not uid or not value:
            continue
        bucket = out.setdefault((acc, uid), [])
        if value not in bucket:
            bucket.append(value)
    return out


# --------------------------------------------------------------------------- #
# 各表读取接口
# --------------------------------------------------------------------------- #
def read_mask_labels(path: str) -> dict[tuple[str, str], list[str]]:
    """``4_masklabel.xlsx`` → ``{(检查号, 序列号): [Maskname, ...]}``。"""
    return _triples(path, value_kws=("maskname", "mask_name", "mask", "掩膜"))


def read_abnormal_labels(path: str) -> dict[tuple[str, str], str]:
    """``1_abnormal.xlsx`` → ``{(检查号, 序列号): Label}``（小写）。

    ``Label`` ∈ {true, fake, compositing, duplicate}：既是"是否异常影像"，
    也指明影像在**哪个子目录**（``true`` → 数据根；其余 → 同名子目录）。
    """
    triples = _triples(path, value_kws=("label", "标签"))
    return {k: v[0].strip().lower() for k, v in triples.items() if v}


def read_duplicate_pairs(path: str) -> list[tuple[str, str]]:
    """``2_duplicate.xlsx`` → ``[(src_img, desc_img), ...]``（正对）。"""
    rows = _rows(path)
    head = _detect_header(rows, ("src_img", "src", "检查号", "accession"))
    if head is None:
        return []
    header = [str(c).strip() for c in rows[head]]
    i_src = _find_col(header, ("src_img", "src", "image1", "检查号1"))
    i_dst = _find_col(header, ("desc_img", "desc", "image2", "检查号2"))
    if i_src is None or i_dst is None:
        print(f"[labels][告警] {os.path.basename(path)} 缺少 src_img/desc_img 列："
              f"{[c for c in header if c][:8]}", flush=True)
        return []
    pairs: list[tuple[str, str]] = []
    for row in rows[head + 1:]:
        src = str(row[i_src]).strip() if i_src < len(row) else ""
        dst = str(row[i_dst]).strip() if i_dst < len(row) else ""
        if src and dst:
            pairs.append((src, dst))
    return pairs


def read_characteristics(path: str,
                         known_ids: set[str] | None = None
                         ) -> dict[str, dict[str, Any]]:
    """``5_characteristics.xlsx`` → ``{检查号: {规范字段: 值}}``。

    列名按官方英文名直读（``Glioma/WHO_grade/Signal_T2WI/…``）；
    二分类存 0/1，``WHO_Grade`` 去掉 Excel 读出来的 ``.0``，其余原样保留
    （``Location`` 可能是 ``|`` 分隔的多标签）。
    """
    rows = _rows(path)
    head = _detect_header(rows, ("accessionnumber", "accession", "检查号", "编号", "id"))
    if head is None:
        print(f"[labels][告警] {os.path.basename(path)} 找不到表头（需要含检查号的列）",
              flush=True)
        return {}
    header = [str(c).strip() for c in rows[head]]
    lower = {c.lower(): i for i, c in enumerate(header) if c}
    i_id = None
    for key in ("accessionnumber", "accession", "检查号", "编号", "id"):
        if key in lower:
            i_id = lower[key]
            break
    if i_id is None:                                              # 退化：包含匹配
        for low, idx in lower.items():
            if "accession" in low or "检查号" in low:
                i_id = idx
                break
    if i_id is None:
        print(f"[labels][告警] {os.path.basename(path)} 认不出检查号列："
              f"{[c for c in header if c][:8]}", flush=True)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in rows[head + 1:]:
        def _cell(idx: int) -> str:
            return str(row[idx]).strip() if idx < len(row) else ""
        acc = _cell(i_id)
        if not acc:
            continue
        record: dict[str, Any] = {}
        for column, field in OFFICIAL_FIELD_COLUMNS.items():
            idx = lower.get(column.lower())
            if idx is None:
                continue
            text = _cell(idx)
            if text == "" or text.lower() in ("nan", "none", "na/unk"):
                continue
            low = text.lower()
            if column in OFFICIAL_BINARY_COLUMNS:
                if low in ("yes", "true", "1", "clear"):
                    record[field] = 1
                elif low in ("no", "false", "0", "unclear"):
                    record[field] = 0
            elif column == "WHO_grade":
                record[field] = text.replace(".0", "")
            else:
                record[field] = text
        if record:
            out[acc] = record
            out[acc.casefold()] = record
    return out


def special_flags(label: str) -> dict[str, float]:
    """官方 ``Label`` → 特殊影像标记（目标一/二-A 的监督信号）。

    ``true`` 表示正常影像；``fake`` 是非人体/伪造，``compositing`` 是拼接，
    ``duplicate`` 是重复影像。三者都以**子目录**形式与主目录同构存放。
    """
    low = (label or "").strip().lower()
    return {
        "fake": 1.0 if low.startswith("fake") else 0.0,
        "stitched": 1.0 if low.startswith(("composit", "stitch")) else 0.0,
        "duplicate": 1.0 if low.startswith("dup") else 0.0,
    }
