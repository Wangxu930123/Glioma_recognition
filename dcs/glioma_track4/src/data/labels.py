"""结构化字段的中文→规范英文枚举映射，以及 csv/xlsx 金标准读取。

规范枚举（《赛事开发规范（赛道四）》prediction.json）：
  Location: Brainstem/Right|Left{Parietal,Frontal,BasalGanglia,Temporal,Cerebellum,Occipital}/Other/NA
  Morphology: Regular/Irregular/NA
  WHO_Grade: 1/2/3/4
  EnhancementPattern: None/Ring/RimEnhancing/Nodular/GroundGlass/Gyriform/Multifocal/Other
  Signal_*: Low/Iso/High
数据集中文（《公共数据集格式说明》赛道4）：
  病灶位置 15 类、病灶形态 3 类、边缘分叶/边界 4 类、坏死/囊变/出血/钙化 3 类、
  T2WI/T2-FLAIR 信号 4 类、T1WI+C 强化 3 类、强化形态 8 类
"""
from __future__ import annotations

import os
import re
from typing import Any

# ---- 位置：中文 → 英文枚举（规范 15 类）----
LOCATION_MAP = {
    "脑干": "Brainstem",
    "右侧顶叶": "RightParietal", "右侧额叶": "RightFrontal",
    "右侧基底节区": "RightBasalGanglia", "右侧颞叶": "RightTemporal",
    "右侧小脑半球": "RightCerebellum", "右侧枕叶": "RightOccipital",
    "左侧顶叶": "LeftParietal", "左侧额叶": "LeftFrontal",
    "左侧基底节区": "LeftBasalGanglia", "左侧颞叶": "LeftTemporal",
    "左侧小脑半球": "LeftCerebellum", "左侧枕叶": "LeftOccipital",
    "其他": "Other", "NA/UNK": "NA", "NA": "NA", "": "NA",
}

# ---- 强化形态：中文 → 规范 8 类 ----
ENHAN_PATTERN_MAP = {
    "多灶状": "Multifocal", "花环状": "RimEnhancing", "环形": "Ring",
    "结节状": "Nodular", "毛玻璃样": "GroundGlass", "脑回状": "Gyriform",
    "其他": "Other", "无": "None", "NA/UNK": "Other", "": "Other",
}

# ---- 形态 ----
MORPH_MAP = {"规则": "Regular", "不规则": "Irregular", "NA/UNK": "NA", "": "NA"}

# ---- 三分类信号：1低/2等/3高 ----
SIGNAL_MAP = {"1低": "Low", "2等": "Iso", "3高": "High", "低": "Low", "等": "Iso", "高": "High",
              "无": "Low", "NA/UNK": None, "": None}

# ---- 有/无（含 4 类变体）----
YESNO_MAP = {"有": 1, "无": 0, "有/清": 1, "无/不清": 0, "无病灶": 0, "NA/UNK": None, "": None}

# ---- 病理结果 → WHO 分级 / 是否胶质瘤 ----
GRADE_MAP = {"脑胶质瘤1级": "1", "脑胶质瘤2级": "2", "脑胶质瘤3级": "3", "脑胶质瘤4级": "4"}
NON_GLIOMA = {"其他肿瘤或病变", "脑转移", "脑脓肿", "脑梗死", "病因不明", "无", "NA/UNK", ""}

# ---- 掩码文件角色识别（《公共数据集格式说明》赛道4 的 ROI 命名）----
MASK_ROLE_KEYWORDS = {
    "core": ["肿瘤瘤体", "瘤体", "core", "et", "enhanc", "增强"],
    "peri": ["全肿瘤", "水肿", "whole", "edema", "flair", "异常信号", "abnormal", "abn"],
    "abn": ["异常信号", "abnormal", "abn"],
}

# ---- 序列模态识别（文件名/序列描述关键词）----
MODALITY_KEYWORDS = {
    "t1c": ["t1c", "t1ce", "t1_ce", "t1+c", "t1wi+c", "t1_wi+c", "postcontrast", "post_contrast",
            "post contrast", "post", "enhance", "增强", "ce+", "+c", "gd", "t1_mprage_post"],
    "flair": ["flair", "t2flair", "t2_flair", "t2-f", "dark_fluid", "darkfluid", "压水", "tirm"],
    "dwi": ["dwi", "diffusion", "diff", "trace", "epi", "弥散"],
    "adc": ["adc", "apparent_diffusion"],
    "swi": ["swi", "susceptibility", "swan", "t2star", "t2*"],
    "t2": ["t2wi", "t2w", "t2_wi", "t2"],
    "t1": ["t1wi", "t1w", "t1_wi", "t1"],
}


def guess_modality(text: str) -> str | None:
    """从文件名/序列描述猜模态；先判 t1c/flair/dwi（它们也含 t1/t2 子串）。"""
    low = (text or "").lower()
    for key in ("t1c", "flair", "dwi", "adc", "swi", "t2", "t1"):
        for kw in MODALITY_KEYWORDS[key]:
            if kw in low:
                return key
    return None


def guess_mask_role(text: str) -> str | None:
    name = text or ""
    for role in ("core", "peri", "abn"):
        for kw in MASK_ROLE_KEYWORDS[role]:
            if kw in name:
                return role
    return None


def mask_role_for(filename: str, modality: str | None = None) -> str | None:
    """**结合掩码所在序列的模态**判定掩码角色（关键修正）。

    仅看文件名会把 FLAIR/T2 序列目录下的"瘤体"误判为任务A的 core，
    进而把掩码写到错误的空间。规范语义：
    - 任务A(core) = **T1 增强**序列上的"肿瘤瘤体"；
    - 任务B(peri) = **FLAIR/T2** 序列上的"瘤体 ∪ 水肿"，或整体勾画的"全肿瘤"；
    - 非肿瘤性病变的"异常信号"按所在序列归位（FLAIR/T2 → peri）。

    返回 ``core`` / ``peri`` / ``abn`` / ``None``。
    """
    import re

    name = (filename or "")
    low = name.lower()
    has = lambda kws: any(k in name for k in kws)                   # noqa: E731
    tok = lambda kws: any(re.search(rf"(^|[^a-z0-9]){re.escape(k)}([^a-z0-9]|$)", low) # noqa: E731
                          for k in kws)
    is_f = modality in ("flair", "t2")

    if has(["异常信号"]) or tok(["abnormal", "abn"]):
        return "peri" if (modality is None or is_f) else "core"
    if has(["全肿瘤", "水肿"]) or tok(["whole", "edema", "peritumoral"]):
        return "peri"
    if has(["肿瘤瘤体", "增强"]) or tok(["core", "et", "enhancing", "enhancement"]):
        return "core"
    if "瘤体" in name:
        return "peri" if is_f else "core"                          # FLAIR/T2 上的瘤体 → 总异常区
    if (has(["肿瘤", "病灶"]) or tok(["tumor", "lesion", "mask", "roi"])) and modality is not None:
        return "peri" if is_f else "core"
    return None


def to_enum(value: Any, mapping: dict) -> Any:
    """宽松匹配：先精确，再包含匹配（处理 "有/清"、"2高" 之类）。"""
    if value is None:
        return None
    s = str(value).strip()
    if s in mapping:
        return mapping[s]
    for k, v in mapping.items():
        if k and k in s:
            return v
    return None


def read_structured_table(path: str) -> dict[str, dict]:
    """读结构化金标准表（csv 或 xlsx）→ {accession_or_patient: {列名: 值}}。

    多键索引：同时尝试 AccessionNumber / PatientId / record_uuid，取能对上的。
    """
    rows: list[dict] = []
    if path.endswith((".xlsx", ".xls")):
        import pandas as pd
        df = pd.read_excel(path, dtype=str)
        rows = df.fillna("").to_dict("records")
    else:
        import csv
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

    out: dict[str, dict] = {}
    for r in rows:
        r = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in r.items()}
        for key in ("AccessionNumber", "accessionNumber", "accession_number",
                    "PatientId", "patient_id", "record_uuid", "PatientID"):
            if r.get(key):
                out[r[key]] = r
                out[r[key].lstrip("0") or r[key]] = r
    return out


def _find_col(row: dict, keywords: list[str], exclude: list[str] | None = None) -> str | None:
    """列名匹配：精确 → 前缀 → 包含；``exclude`` 用于排除干扰列（如 *_pattern）。

    早期版本直接"包含匹配"会让 ``t1wi_c_enhan`` 命中 ``tumor_t1wi_c_enhan_pattern``，
    导致 Enhancement 字段永远拿不到金标准。
    """
    ex = [e.lower() for e in (exclude or [])]
    lower = {str(k).strip().lower(): k for k in row}
    for kw in keywords:                                            # 1) 精确
        if kw.lower() in lower:
            return lower[kw.lower()]
    for kw in keywords:                                            # 2) 前缀
        for k in row:
            kl = str(k).strip().lower()
            if kl.startswith(kw.lower()) and not any(e in kl for e in ex):
                return k
    for kw in keywords:                                            # 3) 包含
        for k in row:
            kl = str(k).strip().lower()
            if kw.lower() in kl and not any(e in kl for e in ex):
                return k
    return None


def structured_from_row(row: dict) -> dict:
    """金标准一行 → 规范字段（缺失字段不出现在结果里 → 训练时自动 mask 掉）。"""
    out: dict[str, Any] = {}

    patho = row.get(_find_col(row, ["病理结果", "pathology", "病理"]) or "", "")
    if patho:
        if patho in GRADE_MAP:
            out["WHO_Grade"] = GRADE_MAP[patho]
            out["TumorProbability"] = 1
        elif patho in NON_GLIOMA or any(k in patho for k in ("转移", "脓肿", "梗死", "无")):
            out["TumorProbability"] = 0

    gl = row.get(_find_col(row, ["glioma_with_label", "胶质瘤"]) or "", "")
    if gl in ("是", "否"):
        out.setdefault("TumorProbability", 1 if gl == "是" else 0)

    def _yn(field: str, kws: list[str], exclude: list[str] | None = None) -> None:
        col = _find_col(row, kws, exclude)
        if col:
            v = to_enum(row.get(col), YESNO_MAP)
            if v is not None:
                out[field] = int(v)

    _yn("Enhancement", ["tumor_t1wi_c_enhan", "t1wi_c_enhan", "强化"], exclude=["pattern", "形态"])
    _yn("Necrosis", ["tumor_feature_necrosis", "necrosis", "坏死"])
    _yn("CysticChange", ["tumor_feature_change", "cysts", "cystic", "囊变"])
    _yn("Hemorrhage", ["tumor_feature_hemorrhage", "hemorrhage", "出血"])
    _yn("Calcification", ["tumor_feature_calcification", "calcification", "钙化"])
    _yn("Lobulation", ["lesion_mor_feature_lobulation", "lobulation", "分叶"])
    _yn("Margin", ["lesion_morph_feature_boundary", "boundary", "边界"])

    for field, kws, mp, ex in (
            ("Morphology", ["lesion_morphology", "病灶形态"], MORPH_MAP, ["feature"]),
            ("Signal_T2WI", ["tumor_t2wi_signal_intensity", "t2wi_signal", "t2信号"], SIGNAL_MAP, []),
            ("Signal_FLAIR", ["tumor_t2_flair_sign_intensity", "flair_sign", "flair信号"], SIGNAL_MAP, []),
            ("EnhancementPattern", ["tumor_t1wi_c_enhan_pattern", "enhan_pattern", "强化形态"],
             ENHAN_PATTERN_MAP, []),
            ("Location", ["location_of_lesion", "病灶位置"], LOCATION_MAP, [])):
        col = _find_col(row, kws, ex)
        if col:
            v = to_enum(row.get(col), mp)
            if v is not None:
                out[field] = v
    return out


def find_structured_tables(root: str) -> list[str]:
    """在数据根下找结构化金标准表（csv/xlsx）。"""
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith((".csv", ".xlsx", ".xls")) and not fn.startswith("~$"):
                hits.append(os.path.join(dirpath, fn))
    return sorted(hits)
