"""官方标注表读取（依据天坛医院参考实现 ``AIRecongition/``）。

官方把标注放在**工程目录**的 ``labels/`` 下（``paths.labels_dir: ./labels``），
是 5 个独立 xlsx：

| 文件 | 关键列 | 用途 |
|---|---|---|
| ``1_abnormal.xlsx`` | AccessionNumber, SeriesUid, **Label** ∈ {true,fake,compositing,duplicate} | 目标一（真实性）/ 目标二-A（拼接）的正样本，**逐序列** |
| ``2_duplicate.xlsx`` | src_img, desc_img | 目标二-B（重复影像）的正对 |
| ``3_serieslabel.xlsx`` | AccessionNumber, SeriesUid, **SeriesLabel** ∈ {T1CE,T2,FLAIR} | **模态的唯一来源** |
| ``4_masklabel.xlsx`` | AccessionNumber, SeriesUid, **Maskname** | 掩膜文件名（任意名，关键词认不出） |
| ``5_characteristics.xlsx`` | AccessionNumber + 14 个英文列 | 目标三/四的结构化字段金标准 |

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
OFFICIAL_LABEL_FILES = {
    "abnormal": "1_abnormal.xlsx",
    "duplicate": "2_duplicate.xlsx",
    "series": "3_serieslabel.xlsx",
    "mask": "4_masklabel.xlsx",
    "characteristics": "5_characteristics.xlsx",
}

#: 环境变量（对应官方 config 的 ``paths.labels_dir``）
LABELS_DIR_ENV = "GLIOMA_LABELS_DIR"

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
def find_official_labels(root: str | os.PathLike,
                         labels_dir: str | os.PathLike | None = None) -> dict[str, str]:
    """定位官方 5 张标注表 → ``{用途: 路径}``（找不到的键不出现）。

    搜索顺序：显式 ``labels_dir`` → ``$GLIOMA_LABELS_DIR`` → **工程目录下的
    ``labels/``**（官方默认 ``./labels``）→ 数据根及其上两级。
    """
    cands: list[Path] = []
    if labels_dir:
        cands.append(Path(labels_dir).expanduser())
    if os.environ.get(LABELS_DIR_ENV):
        cands.append(Path(os.environ[LABELS_DIR_ENV]).expanduser())
    cands.append(Path(__file__).resolve().parents[2] / "labels")      # <工程>/labels
    base = Path(str(root)).expanduser()
    try:
        base = base.resolve()
    except OSError:
        base = base.absolute()
    cands.extend([base, base.parent, base.parent.parent])

    found: dict[str, str] = {}
    for kind, name in OFFICIAL_LABEL_FILES.items():
        for folder in cands:
            candidate = folder / name
            if candidate.is_file():
                found[kind] = str(candidate)
                break
    return found


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
# 五张表各自的读取接口
# --------------------------------------------------------------------------- #
def read_series_labels(path: str) -> dict[tuple[str, str], str]:
    """``3_serieslabel.xlsx`` → ``{(检查号, 序列号): SeriesLabel}``（T1CE/T2/FLAIR）。"""
    triples = _triples(path, value_kws=("serieslabel", "series_label",
                                        "seriestype", "序列类型", "模态"))
    return {k: v[0] for k, v in triples.items() if v}


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
