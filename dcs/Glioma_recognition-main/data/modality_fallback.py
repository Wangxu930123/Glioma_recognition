"""模态识别兜底：**挑不出任何序列时**，改用官方 ``3_serieslabel.xlsx`` 重新贴模态。

为什么需要它
------------
提交侧原有的模态来源是「数据根 ``SeriesType.xlsx`` → 同名 sidecar → 目录名猜关键词」
（见 :mod:`data.loader`）。官方数据下这三样**都可能落空**：

* ``SeriesType.xlsx`` 不随数据集下发（它是我们早期按团队 README 猜的名字）；
* 序列目录名是 DICOM UID（``2.25.135...``），任何关键词都命中不了；
* sidecar 也不保证带 ``SeriesDescription`` / ``ProtocolName``。

三者同时落空时 ``select`` 挑不出序列 → 上层抛「无任何可用序列」（规范 §9.1
不可降级错误）→ **整批评测失败**。而官方 5 张标注表里的 ``SeriesLabel`` 列是
权威模态来源，它躺在团队持久化工作区（如 ``<workspace>/dcs/*/*/labels/``），
**不随数据集下发**，所以必须主动去搜。

工作方式（只在"要报错"时介入，不报错则完全不动）
------------------------------------------------
:func:`data.series_selector.select` 先按原逻辑挑：挑到就**原样返回**——不读盘、
零额外开销、行为与改动前逐字节一致。**只有挑不出任何序列**（= 上层即将抛
「无任何可用序列」或降级推理）时，才调本模块：

1. 定位 ``3_serieslabel.xlsx``（:func:`find_label_table`）：``$GLIOMA_LABELS_DIR``
   → ``<提交工程>/labels`` → ``$COMPETITION_WORKSPACE`` 下 ≤3 层的 ``labels/``
   → 序列文件上溯 4 层的目录（平台有时把表放在数据旁边）；
2. 读表并**在进程内缓存一次**（含"没找到"这一结论，否则每个检查都要重扫工作区）；
3. 两级匹配：``(检查号, 序列号)`` 精确键 → **``SeriesUid`` 单键回退**（官方表里的
   检查号与磁盘目录名口径不一致时，只有 UID 必然一致）；
4. 命中的序列把 ``Series.modality`` 换成官方取值（如 ``T1CE``），交给原逻辑重挑。

只改 ``modality``（描述），**不动** ``metadata``：``series_selector._key_of`` 优先读
``metadata['modality']`` 且**不走关键词匹配**（直接 lower 当键用），把 ``T1CE`` 写进去
会得到 ``t1ce``，反而认不出来。

关掉它：``GLIOMA_MODALITY_FALLBACK=0``（默认开启）。
"""
from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from data.structures import Series, Study

__all__ = [
    "enabled",
    "find_label_table",
    "describe_sources",
    "recover_study",
]

#: 官方序列表文件名（官方仓库写法）；大小写不敏感，另接受 ``*serieslabel*.xlsx`` 变体
_OFFICIAL_NAME = "3_serieslabel.xlsx"

#: 扫工作区时剪掉的目录（缓存/依赖/产物类，钻进去只会白花时间）
_SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".cache", ".git", "site-packages",
    "logs", "log", "runs", "outputs", "checkpoints", "answer", "tmp", "cache",
})

#: 表头别名（归一化后做**子串**匹配，与 ``data.loader._read_series_types`` 同一风格）
_HEADER_ALIASES = {
    "acc": ("accessionnumber", "accession", "检查号", "检查编号"),
    "uid": ("seriesuid", "seriesinstanceuid", "序列号", "序列uid"),
    "lab": ("serieslabel", "seriestype", "序列类型", "模态", "序列描述", "序列名称"),
}

_CACHE: dict[str, Any] = {}


def enabled() -> bool:
    """兜底开关：``GLIOMA_MODALITY_FALLBACK=0`` 可关（默认开启）。"""
    return os.environ.get("GLIOMA_MODALITY_FALLBACK", "").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value if value is not None else "")).casefold()


# --------------------------------------------------------------------------- #
# 表定位
# --------------------------------------------------------------------------- #
def _workspace() -> Path:
    return Path(
        os.environ.get("COMPETITION_WORKSPACE")
        or os.environ.get("WORKSPACE")
        or "/2026aicompetition/workspace"
    ).expanduser()


def _workspace_sweep(max_depth: int = 3) -> list[Path]:
    """``<workspace>`` 下所有名为 ``labels`` 的目录（多份时"更全"者在前）。

    搜索是**有界**的：深度 ≤ ``max_depth``、目录名必须恰好是 ``labels``、剪掉
    缓存/日志类目录，代价与工作区规模无关。结果缓存一次（含空结果）。
    """
    if "ws_dirs" in _CACHE:
        return _CACHE["ws_dirs"]
    ws = _workspace()
    hits: list[Path] = []
    if ws.is_dir():
        def walk(directory: Path, depth: int) -> None:
            try:
                entries = list(os.scandir(directory))
            except OSError:
                return
            for entry in entries:
                try:
                    if not entry.is_dir():
                        continue
                except OSError:
                    continue
                if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
                    continue
                if entry.name == "labels":
                    hits.append(Path(entry.path))          # 命中即止，不再往里钻
                elif depth < max_depth:
                    walk(Path(entry.path), depth + 1)

        walk(ws, 0)

    def score(path: Path) -> tuple[int, float]:
        files = [path / name for name in (_OFFICIAL_NAME, "SeriesType.xlsx")]
        existing = [f for f in files if f.is_file()]
        try:
            newest = max(f.stat().st_mtime for f in existing)
        except ValueError:
            newest = 0.0
        return (len(existing), newest)

    _CACHE["ws_dirs"] = sorted(set(hits), key=score, reverse=True)
    return _CACHE["ws_dirs"]


def _base_dirs() -> list[Path]:
    """与具体检查无关的候选目录（贵的那部分缓存一次）。"""
    if "base_dirs" in _CACHE:
        return _CACHE["base_dirs"]
    dirs: list[Path] = []
    env = os.environ.get("GLIOMA_LABELS_DIR")
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(Path(__file__).resolve().parents[1] / "labels")   # <提交工程>/labels
    dirs.extend(_workspace_sweep())
    _CACHE["base_dirs"] = dirs
    return dirs


def _table_in(folder: Path) -> Path | None:
    exact = folder / _OFFICIAL_NAME
    if exact.is_file():
        return exact
    if folder.is_dir():
        try:                                          # 容忍改名（SeriesLabel.xlsx 等）
            for candidate in sorted(folder.glob("*[Ss]eries[Ll]abel*.xlsx")):
                if candidate.is_file():
                    return candidate
        except OSError:
            return None
    return None


def find_label_table(source_paths: Iterable[Path] = ()) -> Path | None:
    """定位官方序列表；``source_paths`` 用于追加"数据根/父/祖父"候选。"""
    for folder in _base_dirs():
        hit = _table_in(folder)
        if hit:
            return hit
    for raw in source_paths:                          # 平台有时把表放在数据旁边
        try:
            parent = Path(raw).expanduser().resolve().parent
        except OSError:
            continue
        for folder in [parent, *list(parent.parents)[:3]]:
            hit = _table_in(folder)
            if hit:
                return hit
    return None


# --------------------------------------------------------------------------- #
# 表读取
# --------------------------------------------------------------------------- #
def _read_rows(path: Path) -> dict[tuple[str, str], str]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        out: dict[tuple[str, str], str] = {}
        for sheet in workbook.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            header = [_norm(cell) for cell in rows[0]]
            idx: dict[str, int] = {}
            for want, aliases in _HEADER_ALIASES.items():
                for i, cell in enumerate(header):
                    if any(alias in cell for alias in aliases):
                        idx[want] = i
                        break
            if not {"acc", "uid", "lab"} <= idx.keys():
                continue                              # 该 sheet 不是映射表
            for row in rows[1:]:
                try:
                    acc, uid, label = row[idx["acc"]], row[idx["uid"]], row[idx["lab"]]
                except IndexError:
                    continue
                if acc is None or uid is None or label is None:
                    continue
                key = (_norm(acc), _norm(uid))
                value = str(label).strip()
                if not value:
                    continue
                # 同一键冲突：保留**先出现**的取值（官方表偶有重复行，
                # 这里不 fail-fast——兜底模块的职责是尽力救回检查，
                # 而不是让整个评测因为一行脏标注失败；冲突会打进日志）
                if key in out and out[key] != value:
                    print(f"[selector][模态回退] 表中同键冲突 {key}: "
                          f"{out[key]!r} vs {value!r}（保留前者）", flush=True)
                    continue
                out[key] = value
        return out
    finally:
        workbook.close()


def _uid_index() -> dict[str, str]:
    """``{序列号: 官方取值}``（两级匹配的回退索引）+ 表路径，均缓存一次。"""
    if "uid_index" in _CACHE:
        return _CACHE["uid_index"]
    table = find_label_table()
    _CACHE["table"] = table
    index: dict[str, str] = {}
    if table is not None:
        try:
            rows = _read_rows(table)
        except Exception as exc:                       # noqa: BLE001
            print(f"[selector][模态回退] 读官方序列表失败 {table}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows = {}
        for (_, uid), label in rows.items():
            index.setdefault(uid, label)
        print(f"[selector][模态回退] 已读官方序列表 {table.name}："
              f"{len(rows)} 条（UID 单键索引 {len(index)} 条）", flush=True)
    _CACHE["uid_index"] = index
    return index


def describe_sources() -> str:
    """一句话自检"模态来源现在什么状态"（供上层报错文案使用）。"""
    table = _CACHE.get("table")
    if table is None and "uid_index" not in _CACHE:
        table = find_label_table()
        _CACHE["table"] = table
    if table is None:
        return ("未找到官方 3_serieslabel.xlsx（已搜 $GLIOMA_LABELS_DIR、"
                "<提交工程>/labels、$COMPETITION_WORKSPACE 下 3 层、序列文件上溯 4 层）")
    return f"官方 3_serieslabel.xlsx={table}"


def _uid_candidates(series: Series) -> tuple[str, ...]:
    """按可靠性降序：表/元数据里的 UID → 目录名 → 文件名主干。"""
    name = series.source_path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem
    return tuple(dict.fromkeys(
        str(value) for value in (
            series.series_uid,
            series.source_path.parent.name,
            stem,
        ) if value
    ))


def recover_study(study: Study) -> Study:
    """兜底入口：用官方表给"认不出模态"的序列重贴描述。挑不出就原样返回。

    只处理**描述认不出模态**的序列；已有可用描述的序列一律不碰
    （保守起见，避免把本来能用的判断改坏）。
    """
    from data.series_selector import guess_modality

    if not enabled():
        return study
    index = _uid_index()
    if not index:
        if not _CACHE.get("warned_missing"):
            _CACHE["warned_missing"] = True
            print(f"[selector][模态回退] 挑不出序列，且{describe_sources()}；"
                  f"保持原报错（可 export GLIOMA_LABELS_DIR=<表所在目录> 后重跑）",
                  flush=True)
        return study

    accession = study.accession_number
    changed: dict[str, str] = {}
    for series in study.series:
        if guess_modality(series.modality) is not None:
            continue                                   # 原描述已能用 → 不动
        for uid in _uid_candidates(series):
            label = index.get(_norm(uid))
            if label:
                changed[series.series_uid] = label
                break
    if not changed:
        if not _CACHE.get("warned_unmatched"):
            _CACHE["warned_unmatched"] = True
            print(f"[selector][模态回退] 挑不出序列，官方表在 {_CACHE.get('table')} "
                  f"但按 UID 匹配不到（示例 study={accession!r} "
                  f"uid={[s.series_uid for s in study.series][:3]}）", flush=True)
        return study

    recovered = tuple(
        replace(series, modality=changed[series.series_uid])
        if series.series_uid in changed else series
        for series in study.series
    )
    # 同一检查的 select 会被调多次（建体积 → 指纹 → 降级路径），只记一条日志；
    # 这里**不缓存 Study 对象**——它会拖住 Series.image，几千例下来就是内存暴涨。
    logged: set = _CACHE.setdefault("logged", set())
    if accession not in logged:
        logged.add(accession)
        print(f"[selector][模态回退] study {accession!r} 原描述认不出模态，"
              f"已按官方表重贴：{changed}（共 {len(study.series)} 条序列）", flush=True)
    return replace(study, series=recovered)
