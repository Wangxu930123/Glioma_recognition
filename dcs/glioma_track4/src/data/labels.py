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


#: 金标准表里"检查号/病例号"列的关键词（大小写无关；含中文与常见变体）。
#:
#: **顺序即优先级**（精确 → 包含两轮，都按此顺序取第一个命中）：
#: 越具体的检查号命名越靠前，泛化的"记录号/序号"放最后，
#: 避免一张表里同时存在行列号时把行号当成了检查号。
ID_COLUMN_KEYWORDS = ("accessionnumber", "accession_number", "accession_no", "accession",
                      "patientid", "patient_id", "record_uuid", "studyuid",
                      "study_instance_uid", "study_id", "studyid",
                      "检查号", "检查编号", "病例号", "患者号", "检查id",
                      "检查序号", "记录号")


def _find_id_column(row: dict) -> str | None:
    """在表头里定位"检查号"列：**精确 → 包含**两级匹配。

    官方表的表头命名不受我们控制（``AccessionNumber`` / ``检查号`` / ``PatientID``…）。
    早期实现只按几个固定字面量取值，命名一变就整表取不到行 —— 表现为
    ``n_structured_rows: 0`` 且 ``label_field_counts: {}``，
    而表明明就在那儿、内容也齐全，最难查。
    """
    lower = {str(k).strip().lower(): k for k in row}
    for kw in ID_COLUMN_KEYWORDS:
        if kw in lower:
            return lower[kw]
    for kw in ID_COLUMN_KEYWORDS:
        for key_lower, key in lower.items():
            if kw in key_lower:
                return key
    return None


def read_structured_table(path: str) -> dict[str, dict]:
    """读结构化金标准表（csv 或 xlsx）→ {accession_or_patient: {列名: 值}}。

    检查号列按 :data:`ID_COLUMN_KEYWORDS` 自动识别（精确 → 包含），
    因此 ``AccessionNumber`` / ``检查号`` / ``PatientID`` 等命名都能吃。
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
    id_col: str | None = None
    for r in rows:
        r = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in r.items()}
        if id_col is None or id_col not in r:                     # 表头可能换行/换表
            id_col = _find_id_column(r)
        if not id_col:
            continue
        value = r.get(id_col) or ""
        if not value:
            continue
        out[value] = r
        out[value.lstrip("0") or value] = r
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


def _norm_key(value: Any) -> str:
    """归一化查表键（去空白 + 大小写无关）。"""
    return re.sub(r"\s+", "", str(value if value is not None else "")).casefold()


#: 公开别名：序列类型表与其它模块共用同一套键归一化规则，
#: 各写一份迟早会出现"一处去空白、一处不去"的静默错配。
norm_key = _norm_key


#: sidecar JSON 里可能承载序列类型的键（按优先级）
SIDECAR_DESC_KEYS = ("SeriesType", "series_type", "SeriesDescription",
                     "ProtocolName", "SequenceName", "modality", "Modality")

#: **明确的**掩膜线索。判断"这条序列是掩膜"时必须命中其中之一：
#: 类型表里的普通影像名可能带 ``增强``/``et`` 这类词（如 "T1增强"），
#: 而它们同时也是 ``mask_role_for`` 的 core 关键词——不先卡一道，
#: 会把**增强影像**误判成掩膜，于是输入通道里少一个模态、多一个标签。
STRICT_MASK_KW = ("mask", "seg", "label", "roi", "掩码", "标注",
                  "瘤体", "水肿", "异常", "核心", "病灶", "肿瘤区")


def has_strict_mask_hint(text: str) -> bool:
    """该文本是否含**明确**的掩膜线索（大小写无关；中文不受影响）。"""
    low = (text or "").lower()
    return any(k in low for k in STRICT_MASK_KW)

#: 已告警过"发现类型表但缺依赖"（避免每例刷屏）
_WARNED_SERIES_TYPE_DEP = False


def read_series_types(root: str | os.PathLike) -> dict[tuple[str, str], str]:
    """读官方 ``SeriesType.xlsx``：``(检查号, 序列号) → 序列类型``。

    ⚠️ **官方数据下这是模态的唯一来源**。序列目录名是 DICOM UID
    （``1.2.826.0.1...``），靠"按名字猜关键词"一个都命中不了：探针会把整批
    序列归到 ``other``，训练侧则直接报 ``无任何可用序列``——而病例数、目录结构
    看起来完全正常，极易被误判成数据损坏或路径写错。

    与提交工程 ``data/metadata.py`` 同一语义：表头别名容错、缺文件返回空表
    （不是错误）、同一键冲突取值**直接失败**。解析器优先 openpyxl，退化到 pandas。
    """
    global _WARNED_SERIES_TYPE_DEP

    path = os.path.join(str(root), "SeriesType.xlsx")
    if not os.path.isfile(path):
        return {}

    rows: list[list] = []
    try:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in wb.worksheets:
                rows.extend(list(sheet.iter_rows(values_only=True)))
        finally:
            wb.close()
    except ImportError:
        try:
            import pandas as pd
            df = pd.read_excel(path, dtype=str, header=None)
            rows = df.fillna("").values.tolist()
        except Exception as exc:                                  # noqa: BLE001
            if not _WARNED_SERIES_TYPE_DEP:
                _WARNED_SERIES_TYPE_DEP = True
                print(f"[probe][告警] 发现 {path} 但既没有 openpyxl 也没有 pandas，"
                      f"序列类型读不到 → UID 命名的序列会全部归到 other。"
                      f"请 pip install openpyxl（{exc}）", flush=True)
            return {}

    aliases = {
        "acc": ("accessionnumber", "accession", "检查号", "检查编号", "病例号"),
        "uid": ("seriesinstanceuid", "seriesuid", "序列号", "序列uid"),
        "typ": ("seriestype", "type", "序列类型", "模态", "序列描述"),
    }
    out: dict[tuple[str, str], str] = {}
    idx: dict[str, int] = {}
    for row in rows:
        header = [_norm_key(c) for c in row]
        if not idx:
            for want, keys in aliases.items():
                for i, h in enumerate(header):
                    if any(k in h for k in keys):
                        idx[want] = i
                        break
            if set(idx) == {"acc", "uid", "typ"}:
                continue                                          # 表头行本身不入表
            idx = {}
            continue
        try:
            acc, uid, typ = row[idx["acc"]], row[idx["uid"]], row[idx["typ"]]
        except IndexError:
            continue
        if acc in (None, "") or uid in (None, "") or typ in (None, ""):
            continue
        key = (_norm_key(acc), _norm_key(uid))
        value = str(typ).strip()
        if not value:
            continue
        if key in out and out[key] != value:
            raise ValueError(
                f"SeriesType.xlsx 冲突：检查号={acc!r} 序列={uid!r} "
                f"同时映射到 {out[key]!r} 与 {value!r}（{path}）")
        out[key] = value
    return out


def nifti_stem(filename: str) -> str:
    """``x.nii.gz`` / ``x.nii`` → ``x``。"""
    low = filename.lower()
    for ext in (".nii.gz", ".nii"):
        if low.endswith(ext):
            return filename[: -len(ext)]
    return filename


def sidecar_desc(path: str | os.PathLike) -> str | None:
    """读同名 JSON sidecar 的序列描述（不存在或解析失败返回 None）。"""
    import json

    p = os.path.abspath(str(path))
    stem = nifti_stem(os.path.basename(p))
    sidecar = os.path.join(os.path.dirname(p), stem + ".json")
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:                                             # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    for key in SIDECAR_DESC_KEYS:
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return None


#: 不是"结构化金标准"的表（按文件名排除）
_NON_LABEL_TABLE_KW = ("seriestype", "series_type", "gold", "duplicate", "folds")


def find_structured_tables(root: str, max_parents: int = 2) -> list[str]:
    """在数据根（及其**上级 1~2 层**）找结构化金标准表（csv/xlsx）。

    为什么要向上看：官方数据的层级通常是 ``<数据集根>/<某层>/<检查号>/``，
    而字段金标准表常常放在**检查号那一层的上一级**。只扫数据根会出现
    "表明明存在、却一条都没读进来"，报告里只显示 ``{}``，
    分不清是"没表"还是"数据根定位偏了一层"。

    排除两类同名表：``SeriesType.xlsx``（有专用读取器）与 ``gold*.csv``
    （重复影像的金标准，是 pair 列表而非字段表）—— 它们会污染"解析出 N 行"
    这个计数，把诊断信息带偏。
    """
    roots: list[str] = [os.path.abspath(root)]
    parent = roots[0]
    for _ in range(max(0, int(max_parents))):
        parent = os.path.dirname(parent)
        if not parent or parent == os.sep or not os.path.isdir(parent):
            break
        roots.append(parent)

    hits: set[str] = set()
    for idx, base in enumerate(roots):
        if idx == 0:                                              # 数据根：整棵树
            for dirpath, _dirs, files in os.walk(base):
                hits.update(os.path.join(dirpath, fn) for fn in files)
        else:                                                     # 上级：只看本层
            try:
                hits.update(os.path.join(base, fn) for fn in os.listdir(base))
            except OSError:
                continue

    out: list[str] = []
    for path in sorted(hits):
        fn = os.path.basename(path)
        if not fn.endswith((".csv", ".xlsx", ".xls")) or fn.startswith("~$"):
            continue
        if any(k in fn.lower() for k in _NON_LABEL_TABLE_KW):
            continue
        out.append(path)
    return out
