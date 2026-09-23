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

from ..utils.config import data_source_tag, load_paths, resolve
from .labels import (find_structured_tables, guess_modality, has_strict_mask_hint,
                     mask_role_for, norm_key, read_series_types,
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
    for cand in (os.path.join(root, "annotation"), root):
        if os.path.isdir(os.path.join(cand, "Composition")) or os.path.isdir(os.path.join(cand, "fake")):
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

    for cls, key in (("Composition", "composition"), ("fake", "fake"), ("duplicate", "duplicate")):
        d = os.path.join(ann, cls)
        if os.path.isdir(d):
            items = _identifiers(d)
            out[key] = items[:500]
            out[f"{key}_cases"] = items[:500]

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
def _collect_nifti(cdir: str, accession: str = "",
                   series_types: dict | None = None) -> tuple[dict, dict]:
    """扫描一个检查目录下的 NIfTI：返回 (images, mask_entries)。

    - images: ``{modality: {"path","series_uid","file"}}``
    - mask_entries: ``[(role, modality, path, series_uid), ...]``（同一角色可多条 → 取并集）

    ``series_types`` 是官方 ``SeriesType.xlsx`` 解析出的
    ``{(检查号, 序列号): 序列类型}``。**官方数据必须靠它**：序列目录名是
    DICOM UID、文件名也是 UID，任何"按名字猜模态/掩膜"的关键词都命中不了——
    探针会表现为"病例数正常、模态全是 other、掩膜一个没有"，
    而训练侧更直接：``无任何可用序列``。
    """
    images: dict[str, dict] = {}
    masks: list[tuple[str, str | None, str, str]] = []
    for dirpath, _dirs, files in os.walk(cdir):
        sdir = os.path.basename(dirpath)
        for fn in sorted(files):
            if not _is_img(fn) or any(k in fn.lower() for k in SKIP_NAME_KW):
                continue
            full = os.path.join(dirpath, fn)
            stem = _stem(fn)
            series_uid = sdir if sdir and sdir != os.path.basename(cdir) else stem
            # 序列类型：类型表（官方主力）→ sidecar → 目录名/文件名（模拟集）
            desc = ""
            if series_types:
                for uid_key in (series_uid, stem):
                    desc = str(series_types.get(
                        (norm_key(accession), norm_key(uid_key))) or "")
                    if desc:
                        break
            desc = desc or (sidecar_desc(full) or "")
            mod = guess_modality(desc) or guess_modality(stem) or guess_modality(sdir)
            is_pure = stem.strip() in PURE_MODALITY_STEMS
            role = None if is_pure else mask_role_for(fn, mod)
            # 类型表给出的掩膜：文件名是 UID，只能靠"类型 + 严格线索"识别。
            # 严格线索必不可少——"T1增强"这类**影像**名里也含"增强"，
            # 直接送进 mask_role_for 会被判成 core 掩膜。
            if role is None and desc and has_strict_mask_hint(desc):
                role = mask_role_for(f"{desc} {fn}", mod)
            if role:
                masks.append((role, mod, full, series_uid))
            else:
                key = mod or "other"
                if key not in images:
                    images[key] = {"path": full, "series_uid": series_uid, "file": fn}
    return images, masks


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
              series_types: dict | None = None) -> list[dict]:
    """扫描真实影像：一级目录 = 检查号；其下收集影像（NIfTI/DICOM）与掩码。

    ``SeriesType.xlsx``（若存在于数据根）会一次性读入并用于识别**模态与掩膜**：
    官方数据的序列目录名与文件名都是 UID，只靠关键词会得到"整批 other、
    掩膜全无"，而目录结构看起来完全正常。
    """
    cases: list[dict] = []
    if not os.path.isdir(root):
        return cases
    assert_case_root(root)
    if series_types is None:
        series_types = read_series_types(root)
    if series_types:
        print(f"[probe] 已读取 SeriesType.xlsx：{len(series_types)} 条序列类型映射", flush=True)
    entries = sorted(e for e in os.listdir(root)
                     if os.path.isdir(os.path.join(root, e)) and e.lower() != "annotation")
    for acc in entries:
        cdir = os.path.join(root, acc)
        images, mask_entries = _collect_nifti(cdir, acc, series_types)
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
            for key in (acc, acc.lstrip("0"), acc.zfill(len(acc))):
                if key in struct_tables:
                    labels = structured_from_row(struct_tables[key])
                    break
        cases.append({"accession": acc, "dir": cdir, "images": images,
                      "masks": masks, "labels": labels})
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
    for cls, key in (("fake", "fake_cases"), ("Composition", "composition_cases")):
        base = os.path.join(ann, cls)
        for ident in (special.get(key) or []):
            if ident in by:
                continue
            d = os.path.join(base, str(ident))
            if not os.path.isdir(d):
                continue
            imgs, masks = _collect_nifti(d, str(ident), series_types)
            if not imgs:
                imgs = _collect_dicom(d, log)
            if not imgs:
                continue
            c = {"accession": ident, "dir": d, "images": imgs, "masks": {},
                 "labels": {}, "special": cls}
            cases.append(c)
            by[ident] = c
    return cases


def probe(root: str, limit_cases: int | None = None, sample_geometry: int = 8) -> dict:
    log: list[str] = []
    # "按取值找检查号列"需要磁盘上真实存在的检查号（列名叫什么都不影响），
    # 这里先轻量列一次目录名，口径与 scan_real 一致（一级子目录、排除 annotation）。
    known_ids = ({str(e) for e in os.listdir(root)
                  if os.path.isdir(os.path.join(root, e)) and e.lower() != "annotation"}
                 if os.path.isdir(root) else set())
    tables = find_structured_tables(root)
    struct = {}
    for t in tables:
        try:
            struct.update(read_structured_table(t, known_ids=known_ids))
        except Exception:                                         # noqa: BLE001
            pass
    # 序列类型表只读一次，影像与特殊影像两条分支共用（避免"一处读了、一处没读"
    # 导致两边口径不一致）
    # 字典里同一行会有多个键（原值 / 去前导零 / 大小写折叠），
    # 直接 len() 会把"行数"报成实际的两倍以上，把诊断带偏 —— 按**唯一记录**计数。
    n_struct_rows = len({id(v) for v in struct.values()})

    series_types = read_series_types(root)
    special = scan_special(root)
    cases = scan_real(root, limit_cases, struct, log, series_types)
    cases = merge_special_cases(cases, special, log, series_types)

    mod_counter, mask_counter, label_counter = Counter(), Counter(), Counter()
    geom_samples = []
    for c in cases:
        mod_counter.update(c["images"].keys())
        mask_counter.update(c["masks"].keys())
        label_counter.update(c["labels"].keys())
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
    elif not tables:
        labels_hint = ("数据根及其上级 1~2 层都没找到 csv/xlsx 金标准表；"
                       "若表在别处，请把数据根定到与它同级的那一层")
    elif not struct:
        labels_hint = (f"找到 {len(tables)} 个表但一行都没解析出来："
                       f"表里需要有 检查号/AccessionNumber/PatientId 之类的列"
                       f"（候选：{', '.join(os.path.basename(t) for t in tables[:3])}）")
    else:
        labels_hint = (f"表解析出 {n_struct_rows} 行，但列名没映射到规范字段；"
                       f"需要 病理结果 / location_of_lesion / lesion_morphology / "
                       f"tumor_t2wi_signal_intensity 这类列")

    report = {
        "root": os.path.abspath(root),
        "n_cases": len(cases),
        "structured_tables": tables,
        "n_structured_rows": n_struct_rows,
        # 序列类型表命中数：0 且在官方数据上 → 模态/掩膜必然认不出，
        # 先解决这个再谈训练（"病例数正常但全 other"就是这个原因）
        "series_type_rows": len(series_types),
        "modality_counts": dict(mod_counter),
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
            "data_source": data_source_tag(root, phase="train")}


def main() -> None:
    ap = argparse.ArgumentParser()
    paths = load_paths()
    ap.add_argument("--root", default=os.environ.get("DATASET_ROOT") or paths["raw"]["track4"])
    ap.add_argument("--out", default=paths["manifest"])
    ap.add_argument("--limit-cases", type=int, default=None)
    a = ap.parse_args()

    res = probe(a.root, a.limit_cases)
    out = resolve(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"cases": res["cases"], "special": res["special"], "report": res["report"],
                   # 数据源标识：训练前会与本机数据源比对，防止"用本地/公开数据清单训练官方数据"
                   "data_source": res.get("data_source") or data_source_tag(a.root, phase="train"),
                   "data_root": os.path.abspath(a.root)},
                  f, ensure_ascii=False, indent=1)

    print(json.dumps(res["report"], ensure_ascii=False, indent=1))
    print(f"\n[probe] manifest -> {out}  病例 {res['report']['n_cases']}")
    if res["report"]["n_cases"] == 0:
        print("[probe] ⚠️ 未找到病例：请确认 --root 指向含'检查号目录'的数据根（其内应有 NIfTI 或 DICOM）")
    if not res["report"]["label_field_counts"]:
        # 目标三/目标四的监督信号全在这里；为 0 就意味着分类头学不到东西，
        # 而训练照样能跑完（loss 只统计有 mask 的样本）——必须显式提醒。
        print(f"[probe] ⚠️ 结构化字段金标准为空（label_field_counts={{}}）："
              f"{res['report']['labels_hint']}")
    if res["report"]["modality_counts"].get("other") and not res["report"]["series_type_rows"]:
        print("[probe] ⚠️ 有序列落到 other 且没读到 SeriesType.xlsx："
              "官方数据的模态要靠它，见 docs/DATASET_ROOT_TROUBLESHOOT.md")


if __name__ == "__main__":
    main()
