"""数据探针：扫描比赛训练集/测评集 → 自动识别结构并生成 manifest。

识别内容（不依赖任何预设目录布局）：
1. 特殊影像目录 ``annotation/{Composition,fake,duplicate}`` 与重复影像金标准（每行 ``src,desc``）；
2. 真实影像：一级目录 = 检查号；其下按序列分目录或直接放影像；
   **同时支持 NIfTI 与 DICOM**（DICOM 自动转 NIfTI 并缓存，见 ``data/dicom.py``）；
3. 掩码角色：**按掩码所在序列的模态判定**（core=T1C 上的肿瘤瘤体；peri=FLAIR/T2 上的
   瘤体∪水肿 或全肿瘤），同一角色多个掩码取并集；
4. 结构化金标准表（csv/xlsx）→ 规范字段；
5. 抽样读取 shape/spacing/orientation（QC 用）。

用法：
    python -m src.data.probe --root <数据根> [--out data/manifest.json] [--limit-cases 200]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from ..utils.config import data_source_tag, load_paths, resolve, val_root
from .labels import (SERIES_TYPE_TABLE, build_uid_index, find_named_table,
                     find_official_labels, find_structured_tables,
                     guess_modality, has_strict_mask_hint, id_key, is_explicit_other,
                     lookup_series_type, mask_role_for, read_abnormal_table,
                     read_duplicate_pairs, read_mask_table, read_series_types,
                     read_structured_table, sidecar_desc, structured_from_row)

IMG_EXT = (".nii.gz", ".nii")
SKIP_NAME_KW = ("dicomdir", "license", "readme", "vht", ".mhd")

#: 平台 ``/2026aicompetition/datasets`` 下的阶段目录名。
#: 数据根必须精确到其中一个（训练用 ``training``），不能停在父目录。
_PLATFORM_PHASES = frozenset({
    "training", "evaluation_first", "evaluation_second",
    "evaluation_finals", "verification",
})


def assert_case_root(root: str | os.PathLike) -> None:
    """拦截"数据根误指向 ``datasets/`` 父目录"。

    ``/2026aicompetition/datasets`` 下是 5 个阶段目录（``training`` /
    ``evaluation_first`` / ``evaluation_second`` / ``evaluation_finals`` /
    ``verification``），数据根要精确到其中一个。误传父目录时旧实现会把阶段名
    当成检查号：

    * 清单里出现 5 个假病例（``evaluation_first``…），把 5 份数据的像素混在一起；
    * 金标准一张也对不上，``no_labels`` 全空，分类头实际从未收到有效监督；
    * 全程不报错 —— 训练能跑完、指标能打印，直到提交才发现完全跑偏。

    因此直接失败，并在报错里给出应该填的路径。
    """
    if not os.path.isdir(root):
        return
    children = sorted(e for e in os.listdir(root)
                      if os.path.isdir(os.path.join(root, e)))
    if len(children) < 2 or not {c.casefold() for c in children} <= _PLATFORM_PHASES:
        return
    raise ValueError(
        f"数据根 {root} 指向数据集父目录，其下是平台阶段目录 {children}。"
        f"请把数据根设为具体阶段（训练应为 {os.path.join(root, 'training')}）；"
        f"否则这些目录名会被当作检查号，训练数据完全错误却不报错。"
    )
#: 纯模态名（这些是**影像**而非掩码；避免 "flair.nii.gz" 被误判成掩码）
PURE_MODALITY_STEMS = {"t1c", "t1ce", "t1_ce", "t1", "t1w", "t1wi", "t2", "t2w", "t2wi",
                       "flair", "t2flair", "t2_flair", "dwi", "adc", "swi", "bold", "seg",
                       "image", "img", "volume", "scan", "series", "mri", "brain"}


def _is_img(fn: str) -> bool:
    return fn.lower().endswith(IMG_EXT)


def _stem(fn: str) -> str:
    low = fn.lower()
    for ext in IMG_EXT:
        if low.endswith(ext):
            return low[: -len(ext)]
    return low


def _probe_nifti(path: str) -> dict:
    try:
        import nibabel as nib
        img = nib.load(path)
        return {"shape": list(img.shape),
                "spacing": [round(float(x), 3) for x in img.header.get_zooms()[:3]],
                "orient": "".join(nib.aff2axcodes(img.affine))}
    except Exception as e:                                        # noqa: BLE001
        return {"error": str(e)[:80]}


# --------------------------------------------------------------------------- #
# 特殊影像
# --------------------------------------------------------------------------- #
def scan_special(root: str) -> dict:
    """扫描 ``annotation/{Composition,fake,duplicate}`` 与重复影像金标准。

    返回 ``composition`` / ``fake`` 的**病例标识集合**（用于目标一二的监督），
    ``gold_pairs`` 为重复影像金标准对。
    """
    out: dict = {"annotation_dir": None, "composition": [], "fake": [], "duplicate": [],
                 "gold_pairs": [], "composition_cases": [], "fake_cases": []}
    # 目录名两套写法都要认：本地模拟集用 `Composition`，**官方用 `compositing`**
    # （见天坛 `AIRecongition/src/data/paths.py`）。只认前者会让"拼接"这一类
    # 正样本整批找不到 → 目标二-A 的头没有监督信号，而且不报错。
    for cand in (os.path.join(root, "annotation"), root):
        if any(os.path.isdir(os.path.join(cand, name))
               for name in ("Composition", "compositing", "fake")):
            out["annotation_dir"] = cand
            break
    if not out["annotation_dir"]:
        return out
    ann = out["annotation_dir"]

    def _identifiers(d: str) -> list[str]:
        """把特殊影像目录下的条目规整为"病例标识"（目录名，或文件名去扩展名）。"""
        ids = []
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isdir(p):
                ids.append(name)
            elif _is_img(name) or name.lower().endswith((".dcm", ".dicom")):
                ids.append(_stem(name))
        return ids

    # 每个用途可能对应多个目录名（本地 `Composition` / 官方 `compositing`）
    for names, key in ((("Composition", "compositing"), "composition"),
                       (("fake",), "fake"), (("duplicate",), "duplicate")):
        for cls in names:
            d = os.path.join(ann, cls)
            if not os.path.isdir(d):
                continue
            items = _identifiers(d)
            merged = list(dict.fromkeys(out[key] + items))[:500]   # 去重且保序
            out[key] = merged
            out[f"{key}_cases"] = merged

    # 重复影像金标准（csv/txt，每行 src,desc）
    for dirpath, _dirs, files in os.walk(os.path.join(ann, "duplicate")):
        for fn in files:
            if fn.endswith((".csv", ".txt")):
                try:
                    with open(os.path.join(dirpath, fn), encoding="utf-8-sig") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.lower().startswith(("src", "#")):
                                continue
                            parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
                            if len(parts) >= 2:
                                out["gold_pairs"].append([parts[0], parts[1]])
                except Exception:                                 # noqa: BLE001
                    pass
    return out


# --------------------------------------------------------------------------- #
# 真实影像
# --------------------------------------------------------------------------- #
#: 官方数据里"异常影像"的子目录名（`AIRecongition/src/data/paths.py` 约定）：
#: ``<根>/fake/<检查号>/…``、``<根>/compositing/<检查号>/…``、``<根>/duplicate/<检查号>/…``。
#: 它们与主目录**同构**，因此绝不能被当成检查号 —— 否则会多出三个名叫 fake/compositing/
#: duplicate 的"病例"，而它们的"序列"是几百上千个真实病例目录。
SPECIAL_SOURCE_DIRS = ("fake", "compositing", "composition", "duplicate")

#: 允许自动下钻的中间层：``annotation`` / ``original``（影像/标注表所在层）+ 平台阶段名。
#: 平台实际布局比"数据集根"多这一层，见 ``README.md`` §2.2「数据布局」。
#:
#: ``original`` 是**验证集**的实测布局：``verification/original/<检查号>/<序列>/``，
#: 影像、``SeriesType.xlsx`` 与标注表都在 ``verification/original/``。漏了它，
#: 数据根填 ``…/verification`` 时 ``original`` 会被当成检查号 —— 扫描结果是
#: "病例数正常、却报无任何可用序列"，且表也找不到（候选目录里没有它）。
_DESCEND_DIRS = frozenset({"annotation", "original"}) | _PLATFORM_PHASES
#: 顶层非病例目录：本层出现其中任何一个，说明"还没到病例层"
_NON_CASE_DIRS = (frozenset({"annotation", "original", "cache", "runs", "folds",
                             "labels", "logs", "checkpoints", "weights"})
                  | frozenset(SPECIAL_SOURCE_DIRS))


def resolve_case_root(root: str) -> str:
    """把"填高了一层"的数据根下钻到真正含病例目录的那一层。

    平台实测结论（``README.md`` §2.2「数据布局」）：
    ``/2026aicompetition/datasets/training`` 下**只有** ``annotation/``，
    影像、``SeriesType.xlsx`` 与标注表都在 ``training/annotation/`` 里。

    填高一层**不报错**、只静默扫到 0 例 —— 这是最容易踩、也最难查的坑。
    规则与提交工程 ``data/loader.py::_resolve_dataset_root`` 保持一致：
    只有"下一层唯一候选"时才下钻并告警；候选多于一个时**不下钻**，
    交给 :func:`assert_case_root` 报错（猜错阶段比直接失败更糟）。
    """
    if not os.path.isdir(root):
        return root
    if any(os.path.isdir(os.path.join(root, e)) and e.lower() not in _NON_CASE_DIRS
           for e in os.listdir(root)):
        return root                                       # 本层已经有病例目录
    children = sorted(e for e in os.listdir(root)
                      if os.path.isdir(os.path.join(root, e)))
    cands = [c for c in children if c.casefold() in _DESCEND_DIRS]
    if len(cands) == 1:
        sub = os.path.join(root, cands[0])
        print(f"[probe][告警] 数据根 {root} 下没有病例目录，已自动下钻到 "
              f"{cands[0]}/（若不对请用 DATASET_ROOT 显式指定）", flush=True)
        return sub
    return root


def _collect_nifti(cdir: str, accession: str = "",
                   series_types: dict | None = None,
                   mask_names: dict | None = None,
                   uid_index: dict | None = None) -> tuple[dict, list, list]:
    """扫描一个检查目录下的 NIfTI：返回 ``(images, mask_entries, unknown)``。

    - images: ``{modality: {"path","series_uid","file"}}``
    - mask_entries: ``[(role, modality, path, series_uid), ...]``（同一角色可多条 → 取并集）
    - unknown: **认不出模态的序列全表**（``[{...}]``）

    ``unknown`` 为什么必须单独返回：数据的模态来自数据信息 ``SeriesType.xlsx``
    （评测集在正式测试时才随测试数据下发；表没到手或表里没有这个检查号时
    就一条都认不出来），
    序列目录名是 DICOM UID，关键词一个都命中不了 —— 此时唯一的出路是
    **读体素用统计模型判模态**（``data.modality_model``）。而原先的实现
    只把**第一个**认不出的序列塞进 ``images["other"]``、其余直接丢弃，
    兜底模型最多只能看到 1 个序列，且拿不到完整候选。

    不能把列表塞进 ``images``（如 ``images["unknown"] = [...]``）：
    ``inference/writer.py`` 会 ``for mod, meta in images.items()`` 后取
    ``meta["path"]``，遇到 list 会直接崩。

    ``series_types`` 是官方 ``SeriesType.xlsx`` 解析出的
    ``{(检查号, 序列号): 序列类型}``。**官方数据必须靠它**：序列目录名是
    DICOM UID、文件名也是 UID，任何"按名字猜模态/掩膜"的关键词都命中不了——
    探针会表现为"病例数正常、模态全是 other、掩膜一个没有"，
    而训练侧更直接：``无任何可用序列``。
    """
    images: dict[str, dict] = {}
    masks: list[tuple[str, str | None, str, str]] = []
    unknown: list[dict] = []
    for dirpath, _dirs, files in os.walk(cdir):
        sdir = os.path.basename(dirpath)
        for fn in sorted(files):
            if not _is_img(fn) or any(k in fn.lower() for k in SKIP_NAME_KW):
                continue
            full = os.path.join(dirpath, fn)
            stem = _stem(fn)
            series_uid = sdir if sdir and sdir != os.path.basename(cdir) else stem
            # 序列类型：类型表（官方主力）→ sidecar → 目录名/文件名（模拟集）。
            # 类型表里查不到精确键时按 SeriesUid 单键回退：检查号列与磁盘目录名
            # 口径不一致（平台匿名化）时，UID 是两边唯一必然同源的键。
            desc = lookup_series_type(series_types, accession,
                                      (series_uid, stem), uid_index)
            desc = desc or (sidecar_desc(full) or "")
            mod = guess_modality(desc) or guess_modality(stem) or guess_modality(sdir)
            is_pure = stem.strip() in PURE_MODALITY_STEMS
            role = None if is_pure else mask_role_for(fn, mod)
            # 类型表给出的掩膜：文件名是 UID，只能靠"类型 + 严格线索"识别。
            # 严格线索必不可少——"T1增强"这类**影像**名里也含"增强"，
            # 直接送进 mask_role_for 会被判成 core 掩膜。
            if role is None and desc and has_strict_mask_hint(desc):
                role = mask_role_for(f"{desc} {fn}", mod)
            # 官方 `4_masklabel.xlsx` 指定的掩膜：文件名是**任意的**（如 core.nii.gz），
            # 靠关键词认不出。不同步排除的话，掩膜会被当成一路"影像"混进输入通道。
            if role is None and mask_names and fn in (mask_names.get(series_uid) or []):
                role = mask_role_for(f"{fn} {desc}", mod) or "core"
            if role:
                masks.append((role, mod, full, series_uid))
            else:
                meta = {"path": full, "series_uid": series_uid, "file": fn}
                key = mod or "other"
                if key not in images:
                    images[key] = dict(meta)          # 兼容旧语义：仍是"第一个"
                if mod is None and is_explicit_other(desc):
                    # 类型表**明确写了"其他"**（如平台 SeriesType.xlsx 的 其他）：
                    # 这是权威结论"它不是 T1/T2/FLAIR/T1CE 中的任何一个"，
                    # 不是"没认出来"。丢给体素模型猜只会把 DWI/ADC 判成 t2
                    # 填进通道（比空通道更有害），所以标记后不进 unknown。
                    images[key]["declared_other"] = True
                elif mod is None:
                    # 认不出模态的序列**全部保留**（见 docstring：评测期要靠模型回头判）
                    unknown.append({**meta, "reason": desc or stem or sdir})
    return images, masks, unknown


def _collect_dicom(cdir: str, log: list | None = None) -> dict:
    """把 DICOM 序列转成 NIfTI（缓存）→ ``{modality: {...}}``。"""
    from .dicom import ensure_series_nifti
    images: dict[str, dict] = {}
    try:
        recs = ensure_series_nifti(cdir)
    except Exception as e:                                        # noqa: BLE001
        if log is not None:
            log.append(f"{os.path.basename(cdir)}: DICOM 读取失败 {e}")
        return images
    for r in recs:
        mod = guess_modality(r.get("desc") or "") or guess_modality(os.path.basename(
            os.path.dirname(r["path"]))) or guess_modality(r["path"])
        if mod is None:
            continue
        if mod not in images:
            images[mod] = {"path": r["path"], "series_uid": r.get("series_uid") or mod,
                           "file": os.path.basename(r["path"]), "from_dicom": True}
    return images


def scan_real(root: str, limit_cases: int | None = None,
              struct_tables: dict[str, dict] | None = None,
              log: list | None = None,
              series_types: dict | None = None,
              mask_by_acc: dict | None = None) -> list[dict]:
    """扫描真实影像：一级目录 = 检查号；其下收集影像（NIfTI/DICOM）与掩码。

    平台下发的 ``training/annotation/SeriesType.xlsx``（与病例目录**同层**）
    会被一次性读入并用于识别**模态与掩膜**：官方数据的序列目录名与文件名都是
    UID（``2.25.*``），只靠关键词会得到"整批 other、掩膜全无"，
    而病例数与目录结构看起来完全正常。
    """
    cases: list[dict] = []
    root = resolve_case_root(root)                 # 填高一层（如 .../training）时自动下钻
    if not os.path.isdir(root):
        return cases
    assert_case_root(root)
    if series_types is None:
        series_types = read_series_types(root)
    if series_types:
        print(f"[probe] 已读取序列类型映射：{len(series_types)} 条（来源："
              f"{find_named_table(SERIES_TYPE_TABLE, root) or SERIES_TYPE_TABLE}）",
              flush=True)
    # UID 单键回退索引：一次建好、全病例复用（表可能上万行，别放进每病例的循环里）
    uid_index = build_uid_index(series_types)
    entries = sorted(e for e in os.listdir(root)
                     if os.path.isdir(os.path.join(root, e))
                     and e.lower() != "annotation"
                     and e.lower() not in SPECIAL_SOURCE_DIRS)   # fake/compositing/duplicate 是"来源"不是检查号
    skipped_sources = [e for e in sorted(os.listdir(root))
                       if os.path.isdir(os.path.join(root, e))
                       and e.lower() in SPECIAL_SOURCE_DIRS]
    if skipped_sources:
        print(f"[probe] 已按官方约定跳过异常影像目录：{skipped_sources}"
              f"（其内容与主目录同构，按来源区分而非当成检查号）", flush=True)
    for acc in entries:
        cdir = os.path.join(root, acc)
        images, mask_entries, unknown = _collect_nifti(
            cdir, acc, series_types,
            (mask_by_acc or {}).get(acc.casefold()) or (mask_by_acc or {}).get(acc),
            uid_index)
        if not images:                                            # 纯 DICOM 检查
            images = _collect_dicom(cdir, log)
        if not images and not mask_entries:
            continue
        # 掩码 → 角色并集
        masks: dict[str, dict] = {}
        for role, mod, path, uid in mask_entries:
            e = masks.setdefault(role, {"paths": [], "metas": [], "modality": mod})
            e["paths"].append(path)
            e["metas"].append({"path": path, "series_uid": uid, "modality": mod})
        if not struct_tables and not masks and not images:
            continue
        labels = {}
        if struct_tables:
            # 查表要覆盖全部等价写法：原样 / 大小写折叠 / 去前导零 / 归一化键。
            # 只查原样时，目录名 `C0E1F8F2-53BA-45BE` 与表里 `c0e1f8f2-53ba-45be`
            # 互相看不见 —— 表现为"表解析出 N 行，但每例 labels 全空"，
            # 报告里 `label_field_counts: {}` 而 `n_structured_rows` 正常，最难查。
            for key in (acc, acc.casefold(), acc.lstrip("0"),
                        acc.lstrip("0").casefold(), id_key(acc)):
                if key in struct_tables:
                    labels = structured_from_row(struct_tables[key])
                    break
        # unknown_series：认不出模态的序列清单。评测集没有标注表时，
        # 数据集侧（``dataset.pick_series``）会读它们的体素用统计模型判模态。
        cases.append({"accession": acc, "dir": cdir, "images": images,
                      "masks": masks, "labels": labels,
                      **({"unknown_series": unknown} if unknown else {})})
        if limit_cases and len(cases) >= limit_cases:
            break
    return cases


def merge_special_cases(cases: list[dict], special: dict, log: list | None = None,
                        series_types: dict | None = None) -> list[dict]:
    """把 ``annotation/{fake,Composition}`` 中**未出现在真实影像目录**的病例补进清单。

    否则目标一/二的正样本可能一例都匹配不上（``SpecialImageDataset`` 找不到影像），
    特殊影像头仍然训不起来。
    """
    by = {c["accession"]: c for c in cases}
    ann = special.get("annotation_dir")
    if not ann:
        return cases
    uid_index = build_uid_index(series_types)          # UID 单键回退索引（见 _collect_nifti）
    for cls, key in (("fake", "fake_cases"), ("Composition", "composition_cases")):
        base = os.path.join(ann, cls)
        for ident in (special.get(key) or []):
            if ident in by:
                continue
            d = os.path.join(base, str(ident))
            if not os.path.isdir(d):
                continue
            imgs, masks, unknown = _collect_nifti(d, str(ident), series_types,
                                                  uid_index=uid_index)
            if not imgs:
                imgs = _collect_dicom(d, log)
            if not imgs:
                continue
            c = {"accession": ident, "dir": d, "images": imgs, "masks": {},
                 "labels": {}, "special": cls,
                 **({"unknown_series": unknown} if unknown else {})}
            cases.append(c)
            by[ident] = c
    return cases


def probe(root: str, limit_cases: int | None = None, sample_geometry: int = 8,
          phase: str = "train") -> dict:
    log: list[str] = []
    # 先把根定到病例层：否则下面读标注表、列检查号都会落在空的父目录上，
    # 报告里出现"0 例 + 0 张表"，看起来像数据没挂载，实际只是根填高了一层。
    root = resolve_case_root(root)
    # "按取值找检查号列"需要磁盘上真实存在的检查号（列名叫什么都不影响），
    # 这里先轻量列一次目录名，口径与 scan_real 一致（一级子目录、排除 annotation）。
    # 目录名统一归一化后再传：表内取值会经 id_key 归一化，两侧不同口径会
    # 一条都对不上（哈希型检查号 C0E1F8F2-53BA-45BE 就是这么栽的）。
    known_ids = ({id_key(e) for e in os.listdir(root)
                  if os.path.isdir(os.path.join(root, e)) and e.lower() != "annotation"}
                 if os.path.isdir(root) else set())
    # 天坛参考实现那几张表（`1_abnormal` / `2_duplicate` / `4_masklabel` /
    # `5_characteristics`）**不是赛道四数据集的内容**，只在附近有（如团队工作区
    # labels/）时顺手用上：字段金标准、掩膜名表都在这里，靠目录名或关键词猜不出来。
    # 它们的 `3_serieslabel.xlsx` **不再参与**（模态只认数据集自带的 `SeriesType.xlsx`，
    # 上面 read_series_types 已读；工作区那份取值更粗，读了会把 T2WI/T2-Flair 压平）；
    # 数据集里的字段金标准在 `annotation/脑胶质瘤标注结果-训练集.xlsx`
    # （下面 find_structured_tables 会找到）。
    label_files = find_official_labels(root)
    if label_files:
        print("[probe] 官方标注表：" + ", ".join(
            f"{k}={os.path.basename(v)}" for k, v in label_files.items()), flush=True)

    chars = label_files.get("characteristics")                    # 字段金标准（官方列名）
    tables = ([chars] if chars else []) + [t for t in find_structured_tables(root)
                                          if t != chars]
    struct = {}
    for t in tables:
        try:
            struct.update(read_structured_table(t, known_ids=known_ids))
        except Exception as exc:                                  # noqa: BLE001
            # 读表异常**不能静默**：吞掉之后报告里只剩 `label_field_counts: {}`，
            # 看起来像"表里没数据"，实际是解析期就失败了（缺 openpyxl / 文件损坏 /
            # 加密 xlsx）—— 不打印异常就只能靠反复猜。
            print(f"[probe][告警] 读金标准表失败 {os.path.basename(t)}："
                  f"{type(exc).__name__}: {exc}", flush=True)
    # 字典里同一行会有多个键（原值 / 去前导零 / 大小写折叠），
    # 直接 len() 会把"行数"报成实际的两倍以上，把诊断带偏 —— 按**唯一记录**计数。
    n_struct_rows = len({id(v) for v in struct.values()})

    series_types = read_series_types(root)

    # 掩膜：官方 `4_masklabel.xlsx` 的 Maskname（文件名任意，靠关键词认不出）
    mask_by_acc: dict[str, dict[str, list[str]]] = {}
    if label_files.get("mask"):
        for (acc, uid), names in read_mask_table(label_files["mask"]).items():
            slot = mask_by_acc.setdefault(acc.casefold(), {})
            slot[uid] = names
            slot[uid.casefold()] = names
    # 异常影像（fake/compositing/duplicate）的标注来自官方 `1_abnormal.xlsx` 的 Label
    abnormal = (read_abnormal_table(label_files["abnormal"])
                if label_files.get("abnormal") else {})

    special = scan_special(root)
    # 官方重复金标准在 `2_duplicate.xlsx`（两列检查号），不在 `duplicate/` 目录下的 csv；
    # 只扫目录会得到 0 对 → 目标二-B 没有正样本。
    if label_files.get("duplicate"):
        official_pairs = [[a, b] for a, b in read_duplicate_pairs(label_files["duplicate"])]
        if official_pairs:
            special["gold_pairs"] = official_pairs
            print(f"[probe] 已读官方重复金标准 {os.path.basename(label_files['duplicate'])}："
                  f"{len(official_pairs)} 对", flush=True)

    cases = scan_real(root, limit_cases, struct, log, series_types, mask_by_acc)
    cases = merge_special_cases(cases, special, log, series_types)

    mod_counter, mask_counter, label_counter = Counter(), Counter(), Counter()
    geom_samples = []
    n_unknown_series = n_unknown_cases = n_declared_other = 0
    for c in cases:
        mod_counter.update(c["images"].keys())
        mask_counter.update(c["masks"].keys())
        label_counter.update(c["labels"].keys())
        # 认不出模态的序列数：评测集（无标注表、UID 目录名）会整批落在这里。
        # 这是"评测期要不要靠模型判模态"的唯一可见指标 —— 不报出来就只能等
        # 训练时崩「无任何可用序列」才发现。
        n_u = len(c.get("unknown_series") or [])
        n_unknown_series += n_u
        n_unknown_cases += int(n_u > 0)
        # 被类型表**明确标为"其他"**的病例数。它与"认不出"必须分开统计：
        # 前者是权威结论（该排除），后者才需要模型兜底；混在一起会让人
        # 以为"表没接上"，然后跑去重配 labels_dir 白折腾。
        n_declared_other += int(bool((c["images"].get("other") or {}).get("declared_other")))
        if len(geom_samples) < sample_geometry:
            for mod, meta in list(c["images"].items())[:1]:
                geom_samples.append({"accession": c["accession"], "modality": mod,
                                     **_probe_nifti(meta["path"])})

    # 结构化字段金标准为空时**说清是哪一种空**。
    # 三种情形在本地看起来都是 `label_field_counts: {}`，但处理方式完全不同：
    # 没表（数据根偏了一层）/ 有表但认不出检查号列 / 有表有检查号但列名没映射。
    labels_hint = ""
    if label_counter:
        labels_hint = ""
    elif not tables and phase == "val":
        # 官方验证集**没有**字段金标准表（实测 ``verification/original/`` 下只有
        # ``SeriesType.xlsx``）→ 验证集 labels 全空是**预期**，不是配置错。
        # 不说清的话，下面那条"表在别处"的提示会把人引去满磁盘找一张不存在的表。
        labels_hint = ("这是**验证集**：官方验证集实测只有 SeriesType.xlsx、**没有**字段金标准表"
                       " → 本级 labels 全空属**预期**，不必去找表。分类头只由训练折监督；"
                       "官方评估口径是 Dice/NSD/HD95 + 重复影像（scripts/04_eval.sh --split external）。"
                       "若平台后续单独发布验证集标注表：放进验证集目录或 "
                       "export GLIOMA_LABELS_DIR=<含表目录>，重跑本探针即自动接上")
    elif not tables:
        labels_hint = ("数据根/父/祖父、<工程>/labels、$GLIOMA_LABELS_DIR、"
                       "$WORKSPACE 下 3 层都没找到 csv/xlsx 金标准表；"
                       "若表在别处：export GLIOMA_LABELS_DIR=<含表的目录>（或 ln -s 到 "
                       "<工程>/labels），或把数据根定到与它同级的那一层")
    elif not struct:
        labels_hint = (f"找到 {len(tables)} 个表但一行都没解析出来："
                       f"表里需要有 检查号/AccessionNumber/PatientId 之类的列"
                       f"（候选：{', '.join(os.path.basename(t) for t in tables[:3])}）")
    else:
        # 有表、也解析出了行，但**没有一例因此拿到字段**。两种原因的处理方式完全不同：
        # 表属于另一份数据（检查号一条都对不上，例如把训练集的 `5_characteristics.xlsx`
        # 也搜进来了）vs 检查号对上了但列名没映射到规范字段。
        # 只报后一种会把前者说成"列名有问题"，让人反复改列名 —— 先看有没有一行落到磁盘上。
        n_hit = len(known_ids & set(struct))
        if not n_hit:
            labels_hint = (f"表解析出 {n_struct_rows} 行，但与本数据集的检查号**一条都对不上**："
                           f"表多半属于另一份数据/另一个阶段（候选："
                           f"{', '.join(os.path.basename(t) for t in tables[:3])}）——"
                           f"先确认数据根与表是不是配套的")
        else:
            labels_hint = (f"表解析出 {n_struct_rows} 行、命中 {n_hit} 个磁盘检查号，"
                           f"但列名没映射到规范字段；需要 病理结果 / location_of_lesion / "
                           f"lesion_morphology / tumor_t2wi_signal_intensity 这类列")

    report = {
        "root": os.path.abspath(root),
        "n_cases": len(cases),
        "structured_tables": tables,
        "n_structured_rows": n_struct_rows,
        # 官方 5 张标注表的命中情况（看不到某个键 = 那类标注没找到）
        "official_label_files": {k: os.path.basename(v)
                                 for k, v in label_files.items()},
        "n_abnormal_rows": len(abnormal),
        "abnormal_label_counts": dict(Counter(abnormal.values())),
        # 序列类型表命中数：0 且在官方数据上 → 模态/掩膜必然认不出，
        # 先解决这个再谈训练（"病例数正常但全 other"就是这个原因）
        "series_type_rows": len(series_types),
        "modality_counts": dict(mod_counter),
        # 认不出模态的序列总数 / 涉及病例数。评测集没有标注表时它会等于"序列总数"，
        # 此时全靠 data/modality_model.json 兜底（见 README.md §7.2）
        "unknown_series_total": n_unknown_series,
        "cases_with_unknown_series": n_unknown_cases,
        # 类型表里写着"其他"的病例数（`SeriesType.xlsx` 常见）：**不是**缺表信号，
        # 这些序列不参与模态判别（见 labels.EXPLICIT_OTHER_VALUES）
        "cases_with_declared_other_series": n_declared_other,
        "mask_role_counts": dict(mask_counter),
        "label_field_counts": dict(label_counter),
        "labels_hint": labels_hint,
        "special": {k: (v if not isinstance(v, list) else f"{len(v)} items")
                    for k, v in special.items()},
        "geometry_samples": geom_samples,
        "missing_t1c": [c["accession"] for c in cases if "t1c" not in c["images"]][:20],
        "missing_flair": [c["accession"] for c in cases
                          if "flair" not in c["images"] and "t2" not in c["images"]][:20],
        "no_labels": [c["accession"] for c in cases if not c["labels"]][:20],
        "dicom_log": log[:20],
    }
    return {"report": report, "cases": cases, "special": special,
            # phase="val" 时记 `local/<验证集目录名>/val` —— 验证集清单不参与训练，
            # 但同样要可追溯数据来源（评估侧 assert_data_source(phase="val") 会比对）。
            "data_source": data_source_tag(root, phase=phase)}


def main() -> None:
    ap = argparse.ArgumentParser()
    paths = load_paths()
    ap.add_argument("--phase", choices=("train", "val"), default="train",
                    help="train=训练集（默认，写 data/manifest.json）；"
                         "val=官方验证集（写 data/manifest_val.json，"
                         "供评估侧 external 分支使用）")
    ap.add_argument("--root", default=None,
                    help="数据根；train 默认 DATASET_ROOT / raw.track4，"
                         "val 默认 VAL_ROOT / raw.val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-cases", type=int, default=None)
    a = ap.parse_args()

    if a.phase == "val":
        a.root = a.root or val_root()
        if not a.root:
            raise SystemExit(
                "[probe] ✗ 未配置验证集数据根：export VAL_ROOT=<验证集目录>，"
                "或在 configs/paths.yaml 的 raw.val 填写（验证集布局见该处注释）")
        a.out = a.out or paths.get("manifest_val") or "data/manifest_val.json"
    else:
        a.root = a.root or os.environ.get("DATASET_ROOT") or paths["raw"]["track4"]
        a.out = a.out or paths["manifest"]

    res = probe(a.root, a.limit_cases, phase=a.phase)
    out = resolve(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"cases": res["cases"], "special": res["special"], "report": res["report"],
                   # 数据源标识：训练前会与本机数据源比对，防止"用本地/公开数据清单训练官方数据"
                   "data_source": res.get("data_source") or data_source_tag(a.root, phase=a.phase),
                   "data_root": os.path.abspath(a.root)},
                  f, ensure_ascii=False, indent=1)

    print(json.dumps(res["report"], ensure_ascii=False, indent=1))
    _kind = "验证集清单" if a.phase == "val" else "训练清单"
    print(f"\n[probe] {_kind} -> {out}  病例 {res['report']['n_cases']}")
    if a.phase == "val":
        print("[probe] ℹ️ 已生成验证集清单：scripts/04/14/15/16 会自动切到 external 分支"
              "（全折集成，不做留一；最终指标以官方验证集为准）。"
              "想回退折内 val：删除该清单或清空 raw.val/VAL_ROOT。")
    if res["report"]["n_cases"] == 0:
        print("[probe] ⚠️ 未找到病例：请确认 --root 指向含'检查号目录'的数据根（其内应有 NIfTI 或 DICOM）")
    if not res["report"]["label_field_counts"]:
        # 目标三/目标四的监督信号全在这里；为 0 就意味着分类头学不到东西，
        # 而训练照样能跑完（loss 只统计有 mask 的样本）——必须显式提醒。
        # 验证集没有字段金标准表是**预期**（官方只给 SeriesType.xlsx），
        # 用 ⚠️ 会让人以为自己配错了，去翻一张不存在的表。
        _mark = "ℹ️" if a.phase == "val" else "⚠️"
        print(f"[probe] {_mark} 结构化字段金标准为空（label_field_counts={{}}）："
              f"{res['report']['labels_hint']}")
    if res["report"]["modality_counts"].get("other") and not res["report"]["series_type_rows"]:
        # 序列类型表只在**数据集里**（与病例目录同层），走到这里就是没找到 ——
        # 而不是我们没看那几个目录（工作区那份 3_serieslabel.xlsx 已不参与）。
        print(f"[probe] ⚠️ 有序列落到 other 且没读到数据信息表（{SERIES_TYPE_TABLE}）："
              "先确认数据根指向的是含 annotation/ 的那一层（表与病例目录同层），"
              "或 export GLIOMA_LABELS_DIR=<含该表的目录>（/ ln -s 到 <工程>/labels）"
              "再重跑本探针；表也没有时走体素判别兜底（见下一条）。"
              "排查步骤：README.md §7.2")
    if res["report"].get("cases_with_declared_other_series"):
        print(f"[probe] ℹ️ {res['report']['cases_with_declared_other_series']} 例含被类型表标为"
              f"『其他』的序列（不属于 T1/T2-FLAIR/T1CE），已排除、不交给体素模型猜；"
              f"这是正常现象，不必去补标注表")
    if res["report"].get("unknown_series_total"):
        # 评测集没有标注表 → 关键词必然全失效。这条路是**预期**的，
        # 关键是别让它静默：说清有多少路要走模型判别、模型在不在。
        from .modality_model import load_default_model
        has_model = load_default_model() is not None
        print(f"[probe] ℹ️ {res['report']['cases_with_unknown_series']} 例共 "
              f"{res['report']['unknown_series_total']} 路序列模态未知"
              f"（无标注表时的正常现象）→ 体素统计模型："
              f"{'已就绪 data/modality_model.json' if has_model else '❌ 缺失，请先跑 python3 scripts/31_train_modality_model.py --root <数据根>'}")


if __name__ == "__main__":
    main()
