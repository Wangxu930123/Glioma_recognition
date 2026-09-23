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
from pathlib import Path
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


#: 只按**精确相等**匹配的短列名。
#:
#: 不能并进上面的"包含"匹配：``id`` 会命中 ``SeriesUid``，
#: 于是整表按**序列号**建索引 —— 检查号永远对不上、字段全空，
#: 而且它看起来"解析成功了"（有行数、无字段），比认不出更难查。
ID_COLUMN_EXACT = ("id", "编号", "序号", "流水号")

#: 判定"这行是表头"用的字段线索（命中越多越像表头）
FIELD_HINTS = ("病理", "glioma", "location", "lesion", "morpholog", "tumor",
               "signal", "enhan", "坏死", "囊变", "出血", "钙化", "强化", "水肿")

#: 已告警过的表（避免每次探测/训练都刷屏）
_WARNED_TABLES: set[str] = set()


def _find_id_column(header) -> str | None:
    """在**表头单元格序列**里定位"检查号"列，返回命中的列名（原样）。

    入参是表头**各单元格的值**（不是整行、也不是 dict）。这里踩过一次坑：
    传 ``dict(enumerate(row))`` 时键变成了下标，于是"列名"永远匹配不上，
    所有表都解析出 0 行 —— 连本来正常的小写 csv 也一起失效。

    两轮匹配：① 精确（含 ``id``/``编号`` 这类短名）→ ② 包含
    （**只用长关键词**，避免 ``id`` 命中 ``SeriesUid`` 而错把序列号当检查号）。
    """
    cells = [str(c).strip() for c in header if str(c).strip()]
    lower = {c.lower(): c for c in cells}
    for kw in ID_COLUMN_KEYWORDS + ID_COLUMN_EXACT:                # ① 精确
        if kw in lower:
            return lower[kw]
    for kw in ID_COLUMN_KEYWORDS:                                  # ② 包含（不用短名）
        for key_lower, key in lower.items():
            if kw in key_lower:
                return key
    return None


# --------------------------------------------------------------------------- #
# 官方标注表（天坛医院参考实现 `AIRecongition/` 的约定）
# --------------------------------------------------------------------------- #
#: 官方标注表文件名 → 用途
#:
#: | 文件 | 关键列 | 用途 |
#: |---|---|---|
#: | ``1_abnormal.xlsx`` | AccessionNumber, SeriesUid, **Label** ∈ {true,fake,compositing,duplicate} | 目标一/二的正样本（**逐序列**） |
#: | ``2_duplicate.xlsx`` | src_img, desc_img | 重复影像 pair |
#: | ``3_serieslabel.xlsx`` | AccessionNumber, SeriesUid, **SeriesLabel** ∈ {T1CE,T2,FLAIR} | **模态的唯一来源** |
#: | ``4_masklabel.xlsx`` | AccessionNumber, SeriesUid, **Maskname** | 掩膜文件名（任意名，关键词认不出） |
#: | ``5_characteristics.xlsx`` | AccessionNumber + 14 个英文列 | 结构化字段金标准 |
#:
#: 这套约定是**唯一权威**。在此之前我们按 ``SeriesType.xlsx`` / 中文列名去猜，
#: 于是出现"病例数正常、一例都挑不出模态""字段金标准为空"——
#: 表一直都在，只是文件名和列名都不是我们猜的那套。
OFFICIAL_LABEL_FILES = {
    "abnormal": "1_abnormal.xlsx",
    "duplicate": "2_duplicate.xlsx",
    "series": "3_serieslabel.xlsx",
    "mask": "4_masklabel.xlsx",
    "characteristics": "5_characteristics.xlsx",
}

#: 官方标注表所在目录的环境变量（对应官方 config 的 ``paths.labels_dir``）
LABELS_DIR_ENV = "GLIOMA_LABELS_DIR"


#: 在 ``$WORKSPACE`` 下做有界搜索时要跳过的目录（缓存/日志里不可能有官方表，
#: 但它们动辄上万个子目录，不剪掉会让这一步从"秒级"变成"分钟级"）
_SKIP_WORKSPACE_DIRS = frozenset({
    "cache", "cache_nifti", "cache_dicom", "logs", "checkpoints", "runs", "outputs",
    "tmp", "node_modules", "__pycache__",
})


def workspace_labels_dirs(max_depth: int = 3) -> list[Path]:
    """``$WORKSPACE`` 下所有名为 ``labels`` 的目录（按"官方表更全 + 更新"排序）。

    为什么需要它：官方 5 张表**不随数据集下发**，通常躺在团队持久化工作区里
    （如 ``<workspace>/dcs/goal1and2/Goal1and2/labels``）。有了这一步，容器里
    **零配置**就能读到表，不必每个人都记得 ``export GLIOMA_LABELS_DIR``。

    搜索是**有界**的（深度 ≤ ``max_depth``、目录名必须恰好是 ``labels``、
    剪掉缓存/日志类目录），因此代价与工作区规模无关。
    """
    ws = Path(os.environ.get("WORKSPACE") or "/2026aicompetition/workspace").expanduser()
    if not ws.is_dir():
        return []
    hits: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        try:
            entries = list(os.scandir(d))
        except OSError:
            return
        for e in entries:
            try:
                if not e.is_dir():
                    continue
            except OSError:
                continue
            if e.name.startswith(".") or e.name in _SKIP_WORKSPACE_DIRS:
                continue
            child = Path(e.path)
            if e.name == "labels":
                hits.append(child)                      # 命中即止，不再往里钻
                continue
            if depth < max_depth:
                walk(child, depth + 1)

    walk(ws, 0)

    def score(p: Path) -> tuple[int, float]:
        files = [p / name for name in OFFICIAL_LABEL_FILES.values()]
        n = sum(f.is_file() for f in files)
        try:
            mt = max(f.stat().st_mtime for f in files if f.is_file())
        except (OSError, ValueError):
            mt = 0.0
        return (n, mt)                                  # 表更全优先，其次更新

    return sorted(set(hits), key=score, reverse=True)


def find_official_labels(root: str | os.PathLike | None = None,
                         labels_dir: str | os.PathLike | None = None) -> dict[str, str]:
    """定位官方 5 张标注表 → ``{用途: 路径}``（找不到的键不出现）。

    搜索顺序（逐个候选目录试，命中即止）：

    1. 显式传入的 ``labels_dir``（对应官方 config 的 ``paths.labels_dir``）
    2. 环境变量 ``GLIOMA_LABELS_DIR``
    3. **本工程目录下的 ``labels/``**（官方默认 ``./labels``）
    4. ``$WORKSPACE`` 下名为 ``labels`` 的目录（**团队工作区里那份**；平台不随数据集下发，
       见 :func:`workspace_labels_dirs`。容器里靠这一步做到零配置）
    5. 数据根自身、父目录、祖父目录（平台有时把标注放在数据旁边）

    官方把标注放在**工程目录**而不是数据集里 —— 这也是"数据根下找不到金标准"的原因之一。
    ``root=None`` 时只搜 1~4（用于报错时做"表到底在不在"的自检）。
    """
    cands: list[Path] = []
    if labels_dir:
        cands.append(Path(labels_dir).expanduser())
    if os.environ.get(LABELS_DIR_ENV):
        cands.append(Path(os.environ[LABELS_DIR_ENV]).expanduser())
    cands.append(Path(__file__).resolve().parents[2] / "labels")      # <工程>/labels
    cands.extend(workspace_labels_dirs())                             # $WORKSPACE/**/labels
    if root is not None:
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


def _find_col_in_list(header: list[str], keywords: tuple[str, ...]) -> int | None:
    """在表头列表里找列，返回**列号**：精确 → 前缀 → 包含（按关键词顺序）。"""
    lower = {str(c).strip().lower(): i for i, c in enumerate(header) if str(c).strip()}
    for kw in keywords:                                            # ① 精确
        if kw in lower:
            return lower[kw]
    for kw in keywords:                                            # ② 前缀
        for low, idx in lower.items():
            if low.startswith(kw):
                return idx
    for kw in keywords:                                            # ③ 包含
        for low, idx in lower.items():
            if kw in low:
                return idx
    return None


def read_official_triples(path: str,
                          id_kws: tuple[str, ...] = ("accessionnumber", "accession", "检查号"),
                          uid_kws: tuple[str, ...] = ("seriesuid", "series_uid", "序列号"),
                          value_kws: tuple[str, ...] = ("label",),
                          ) -> dict[tuple[str, str], list[str]]:
    """读官方"检查号 + 序列号 + 取值"三类表 → ``{(检查号, 序列号): [取值, ...]}``。

    ``1_abnormal`` / ``3_serieslabel`` / ``4_masklabel`` 都是这个形状
    （掩膜表同一序列可能有多行 → 取值列表）。取值保持原样大小写，
    调用方按需 ``.upper()`` / ``.lower()`` 比较。
    """
    out: dict[tuple[str, str], list[str]] = {}
    for rows in _sheet_rows(path):
        head = _detect_header(rows)
        if head is None:
            continue
        header = [str(c).strip() for c in rows[head]]
        i_id = _find_col_in_list(header, id_kws)
        i_uid = _find_col_in_list(header, uid_kws)
        i_val = _find_col_in_list(header, value_kws)
        if i_id is None or i_uid is None or i_val is None:
            continue
        for row in rows[head + 1:]:
            def _cell(idx: int) -> str:
                return str(row[idx]).strip() if idx < len(row) else ""
            acc, uid, value = _cell(i_id), _cell(i_uid), _cell(i_val)
            if not acc or not uid or not value:
                continue
            for key in {(acc, uid), (acc.casefold(), uid.casefold())}:
                bucket = out.setdefault(key, [])
                if value not in bucket:
                    bucket.append(value)
    return out


def read_serieslabel_table(path: str) -> dict[tuple[str, str], str]:
    """读 ``3_serieslabel.xlsx`` → ``{(检查号, 序列号): SeriesLabel}``（模态）。"""
    triples = read_official_triples(
        path,
        value_kws=("serieslabel", "series_label", "seriestype", "序列类型", "模态", "序列标签"),
    )
    return {k: v[0] for k, v in triples.items() if v}


def read_mask_table(path: str) -> dict[tuple[str, str], list[str]]:
    """读 ``4_masklabel.xlsx`` → ``{(检查号, 序列号): [Maskname, ...]}``。"""
    return read_official_triples(
        path,
        value_kws=("maskname", "mask_name", "mask", "掩膜", "标注文件"),
    )


def read_duplicate_pairs(path: str) -> list[tuple[str, str]]:
    """读 ``2_duplicate.xlsx`` → ``[(src_img, desc_img), ...]``（重复影像正对）。

    官方把重复金标准放在**标注表**里（两列检查号），而不是 ``duplicate/`` 目录下的
    csv —— 只扫目录会得到 0 对，重复任务就没有正样本。
    """
    pairs: list[tuple[str, str]] = []
    src_kws = ("src_img", "src", "image1", "检查号1", "studyuid")
    dst_kws = ("desc_img", "desc", "image2", "检查号2", "studyuid_dup")
    for rows in _sheet_rows(path):
        # ⚠️ 不能复用通用表头识别：它要求"检查号列"，而这张表只有 src/desc 两列，
        # 于是永远返回 None → 重复金标准恒为 0 对（不报错）。
        head = None
        for idx, row in enumerate(rows[:20]):
            header = [str(c).strip() for c in row]
            if (_find_col_in_list(header, src_kws) is not None
                    and _find_col_in_list(header, dst_kws) is not None):
                head = idx
                break
        if head is None:
            continue
        header = [str(c).strip() for c in rows[head]]
        i_src = _find_col_in_list(header, src_kws)
        i_dst = _find_col_in_list(header, dst_kws)
        if i_src is None or i_dst is None or i_src == i_dst:
            continue
        for row in rows[head + 1:]:
            src = str(row[i_src]).strip() if i_src < len(row) else ""
            dst = str(row[i_dst]).strip() if i_dst < len(row) else ""
            if src and dst:
                pairs.append((src, dst))
    return pairs


def read_abnormal_table(path: str) -> dict[tuple[str, str], str]:
    """读 ``1_abnormal.xlsx`` → ``{(检查号, 序列号): Label}``。

    ``Label ∈ {true, fake, compositing, duplicate}``：它同时告诉我们两件事 ——
    这条序列是不是异常影像，以及它的影像在哪个子目录下
    （``true`` → 数据根；``fake``/``compositing``/``duplicate`` → 同名子目录）。
    """
    triples = read_official_triples(path, value_kws=("label", "标签"))
    return {k: v[0].lower() for k, v in triples.items() if v}


def _sheet_rows(path: str) -> list[list[list[str]]]:
    """把 csv/xlsx 读成"若干张表、每张是原始行"（**不做任何表头假设**）。

    为什么不直接用 ``pandas.read_excel`` 的默认行为：它把**第一行**当表头、
    且**只读第一个 sheet**。中文标注表的常见排版是

        A1: 脑胶质瘤标注结果（训练集）      ← 标题
        A2: （空行 / 填表说明）
        A3: 检查号 | 病理结果 | …           ← 真正的表头

    默认行为下列名会变成"标题/Unnamed"，检查号列认不出来，整表 0 行。
    """
    if path.endswith((".xlsx", ".xls")):
        import pandas as pd
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
        return [[["" if v is None else str(v).strip() for v in row]
                 for row in frame.fillna("").values.tolist()]
                for frame in sheets.values()]
    import csv
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [[[str(c).strip() for c in row] for row in csv.reader(f)]]


def _detect_header(rows: list[list[str]], max_scan: int = 20) -> int | None:
    """在前若干行里找**真正的表头行**：必须含检查号列，字段线索越多越优先。

    只看第一行是这个数据最常见的失效点（标题行占了第一行）；
    而"必须含检查号列"这条同时挡住了把说明行、数据行误判成表头。
    """
    best: tuple[int, int] | None = None                            # (线索数, 行号)
    for idx, row in enumerate(rows[:max_scan]):
        if not any(str(c).strip() for c in row):
            continue                                               # 空行
        if _find_id_column(row) is None:
            continue
        hits = sum(1 for c in row
                   if any(k in str(c).lower() for k in FIELD_HINTS))
        if best is None or hits > best[0]:
            best = (hits, idx)
    return best[1] if best else None


def _id_key(value) -> str:
    """把"检查号"归一化成可比较的键：只留字母数字，去前导零，大小写无关。

    这样 ``C0E1F8F2-53BA_45BE``、``c0e1f8f253ba45be``、`` 00123 `` 都能对上。

    另修一个 Excel 专属坑：数字型检查号读出来常带小数尾巴（``1234567.0``），
    若直接去掉非字母数字会拼成 ``12345670`` —— 与目录名 ``1234567``
    差一位、永远不相等。所以先把结尾的 ``.0`` 摘掉再归一化。
    本函数**幂等**：对已归一化的键再调用结果不变。
    """
    text = str(value).strip()
    if "." in text:
        text = re.sub(r"\.0+$", "", text)                           # 1234567.0 → 1234567
    text = re.sub(r"[^0-9a-z]+", "", text.casefold())
    return text.lstrip("0") or text


#: 公开别名：检查号归一化是"探针/数据集/脚本"三方共用的口径，
#: 各写一份必然出现"一处归一化、一处原样"的静默错配（本文件已因此踩过一次）。
id_key = _id_key


def _best_id_column_by_values(rows: list[list[str]], known_keys: set[str],
                              min_hit: float = 0.3, max_scan: int = 300
                              ) -> tuple[int, int] | None:
    """**按取值**找出检查号列：返回 ``(列号, 第一处命中的行号)``。

    这是"不猜列名"的做法：磁盘上已经有哪些检查号是确定的，
    拿它去逐列比对即可 —— 列名写 ``编号``/``AccessionNumber``/``患者ID``
    甚至乱码都不影响。比维护关键词表可靠得多（关键词表每遇到一种新命名就失效一次）。

    ⚠️ ``known_keys`` 允许是**原始目录名**（调用方常直接把 ``os.listdir`` 结果传进来）：
    这里统一做一次 :func:`_id_key` 归一化再比。此前只有表内取值归一化、
    调用方若是原样传入，遇到 ``C0E1F8F2-53BA-45BE`` 这种带大写/连字符的哈希
    检查号就会**一条都对不上**，于是静默退化成"按列名找"→ 整表 0 行。
    ``_id_key`` 幂等，已归一化的调用方重复传也安全。
    """
    if not rows or not known_keys:
        return None
    known_keys = {_id_key(k) for k in known_keys}
    width = max(len(r) for r in rows[:max_scan]) if rows[:max_scan] else 0
    best: tuple[float, int, int] | None = None            # (命中率, 命中数, 列号)
    for col in range(width):
        hit = total = first_hit = 0
        for idx, row in enumerate(rows[:max_scan]):
            value = str(row[col]).strip() if col < len(row) else ""
            if not value:
                continue
            total += 1
            if _id_key(value) in known_keys:
                hit += 1
                if not first_hit:
                    first_hit = idx
        if total >= 3 and hit / total >= min_hit:
            score = (hit / total, hit, col)
            if best is None or score > best:
                best = score
    return (best[2], next(i for i, r in enumerate(rows[:max_scan])
                          if best[2] < len(r) and _id_key(r[best[2]]) in known_keys)) \
        if best else None


def _header_row_above(rows: list[list[str]], data_row: int) -> int | None:
    """数据行往上找**最后一个不像数据**的行当表头（跳过空行）。

    官方表的表头行取值不会是检查号，因此"往上第一个不含检查号样式的行"就是表头。
    """
    for idx in range(data_row - 1, -1, -1):
        row = rows[idx]
        if not any(str(c).strip() for c in row):
            continue                                          # 空行：跳过继续往上
        return idx
    return None


def dump_table(path: str, max_rows: int = 8, max_cols: int = 12,
               cell_width: int = 22) -> str:
    """把表**整个摊开**成可读文本（工作表名、行列数、前若干行）。

    用途很直接：当解析失败时不要让人猜表长什么样 —— 直接打印出来看。
    """
    lines: list[str] = [f"文件：{path}"]
    sheets = _sheet_rows(path)
    lines.append(f"工作表数：{len(sheets)}")
    for s_idx, rows in enumerate(sheets):
        width = max((len(r) for r in rows), default=0)
        lines.append(f"\n--- sheet{s_idx}: {len(rows)} 行 × {width} 列 ---")
        for r_idx, row in enumerate(rows[:max_rows]):
            cells = [str(c)[:cell_width].ljust(cell_width)
                     for c in row[:max_cols]]
            more = " …" if len(row) > max_cols else ""
            lines.append(f"  行{r_idx:>3}: " + " | ".join(cells) + more)
        if len(rows) > max_rows:
            lines.append(f"  ……（还有 {len(rows) - max_rows} 行）")
    return "\n".join(lines)


def _diagnose_id_columns(sheets: list[list[list[str]]], known_keys: set[str],
                         top: int = 3, max_scan: int = 300) -> str:
    """解析失败时给出**可执行的**原因：哪一列最像检查号、命中多少行。

    "0 行"有两种成因，处理方式相反：
    ① 检查号列存在，但取值与磁盘目录名不是同一套编号（要按值映射或换表）；
    ② 表里根本没有检查号（这表不是字段金标准）。
    只看 `label_field_counts: {}` 分不清，只能反复猜 —— 所以把命中率打出来。
    """
    if not known_keys:
        return ("  ⚠️ 取不到磁盘上的检查号（数据根下没有病例目录），"
                "只能按列名识别 —— 请先把数据根指到含检查号目录的那一层。")
    lines: list[str] = []
    for s_idx, rows in enumerate(sheets):
        if not rows:
            continue
        width = max(len(r) for r in rows[:max_scan])
        scored: list[tuple[float, int, int, int]] = []                 # (命中率,命中,非空,列)
        for col in range(width):
            hit = total = 0
            for row in rows[:max_scan]:
                value = str(row[col]).strip() if col < len(row) else ""
                if not value:
                    continue
                total += 1
                if _id_key(value) in known_keys:
                    hit += 1
            if total >= 3 and hit:
                scored.append((hit / total, hit, total, col))
        scored.sort(key=lambda s: (-s[0], -s[1]))
        if not scored:
            head = [str(c)[:18] for c in rows[min(1, len(rows) - 1)][:8]]
            lines.append(f"  sheet{s_idx}: 没有哪一列的取值能对上磁盘检查号"
                         f"（前几列表头={head}）→ 这表多半不是字段金标准")
            continue
        for ratio, hit, total, col in scored[:top]:
            name = ""
            for r in rows[:20]:                                        # 该列的表头文字
                if col < len(r) and str(r[col]).strip() and _id_key(r[col]) not in known_keys:
                    name = str(r[col]).strip()[:18]
                    break
            lines.append(f"  sheet{s_idx}: 第 {col} 列最像检查号（表头 {name!r}）"
                         f"命中 {hit}/{total} 行（{ratio:.0%}）")
        lines.append(f"  → 上面命中率若明显低于 30%，说明该表编号与目录名不是同一套；"
                     f"把 dump 出来的前几行发出来即可确定映射关系")
    return "\n".join(lines)


def read_structured_table(path: str, known_ids: set[str] | None = None) -> dict[str, dict]:
    """读结构化金标准表（csv 或 xlsx）→ {检查号: {列名: 值}}。

    三个"看起来应该没问题、实际常常出问题"的地方都做了处理：

    1. **标题行/空行**：中文标注表的排版通常是
       ``标题行 → 空行/说明 → 真正的表头``，而 ``pandas.read_excel``
       默认把第一行当表头 —— 列名成了"标题/Unnamed"，检查号列认不出来，
       整表解析出 **0 行**。这里改为自己在前若干行里找表头行。
    2. **多工作表**：默认只读第一个 sheet；这里遍历全部 sheet。
    3. **检查号大小写**：目录名是小写哈希、表里可能是大写，
       因此大小写折叠后的键也一并登记。

    一行都没解析出来时会打印告警（含表头预览），
    而不是只给上层返回一个空字典。
    """
    out: dict[str, dict] = {}
    sheets = _sheet_rows(path)
    for rows in sheets:
        header: list[str] = []
        data_start: int | None = None
        id_col: int | None = None

        # ① 首选：**按取值**找检查号列（不依赖列名，见 _best_id_column_by_values）
        if known_ids:
            found = _best_id_column_by_values(rows, known_ids)
            if found is not None:
                id_col, first_data = found
                head = _header_row_above(rows, first_data)
                if head is not None:
                    header = [str(c).strip() for c in rows[head]]
                data_start = first_data

        # ② 退化：按列名找（离线单独解析、或表里本来就没有磁盘上的检查号时）
        if data_start is None:
            head = _detect_header(rows)
            if head is None:
                continue
            header = [str(c).strip() for c in rows[head]]
            id_name = _find_id_column(header)
            if not id_name:
                continue
            id_col = header.index(id_name)
            data_start = head + 1

        for row in rows[data_start:]:
            if not any(str(c).strip() for c in row):
                continue                                          # 跳过空行
            record: dict[str, str] = {
                col: (str(row[i]).strip() if i < len(row) else "")
                for i, col in enumerate(header) if col
            }
            if not record:                                        # 表头认不出：用列号占位
                record = {f"col{i}": (str(row[i]).strip() if i < len(row) else "")
                          for i in range(len(row))}
            value = str(row[id_col]).strip() if id_col < len(row) else ""
            if not value:
                continue
            stripped = value.lstrip("0") or value
            # 登记**归一化键**：目录名可能是 `C0E1F8F2-53BA-45BE`，表里是
            # `c0e1f8f253ba45be`（或反之），只有原始形式会一条都查不到。
            for key in {value, stripped, value.casefold(), stripped.casefold(),
                        _id_key(value)}:
                out[key] = record

    if not out and path not in _WARNED_TABLES:
        _WARNED_TABLES.add(path)
        diag = _diagnose_id_columns(sheets, {_id_key(k) for k in (known_ids or ())})
        print(f"[labels][告警] {os.path.basename(path)} 未解析出任何行。\n"
              f"  已尝试：按取值比对检查号列 → 按列名识别 → 跳过标题行 → 遍历全部工作表。\n"
              + (diag + "\n" if diag else "")
              + f"  下面把表整个摊开，直接看它长什么样：\n{dump_table(path)}", flush=True)
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


#: 官方 ``5_characteristics.xlsx`` 的列名 → 我们的规范字段
#: （值域见天坛参考实现 ``src/tasks/characteristics/schema.py``）
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

#: 官方列里属于**二分类**的（存成 0/1，与 team 侧 FIELD_ENUMS 一致）
OFFICIAL_BINARY_COLUMNS = {"Glioma", "Enhancement", "Necrosis", "CysticChange",
                           "Hemorrhage", "Calcification", "Margin", "Lobulation"}


def _official_columns_to_fields(row: dict) -> dict[str, Any]:
    """按**官方英文列名**直接映射（权威路径，不猜中文关键词）。

    官方 ``5_characteristics.xlsx`` 的 14 列是
    ``Glioma / WHO_grade / Enhancement / EnhancementPattern / Necrosis /
    CysticChange / Hemorrhage / Calcification / Margin / Lobulation /
    Morphology / Signal_T2WI / Signal_FLAIR / Location``，
    与模拟集的中文列名是两套东西。之前只写中文关键词，官方表自然一条也映射不出来。
    """
    lower = {str(k).strip().lower(): k for k in row}
    out: dict[str, Any] = {}
    for column, field in OFFICIAL_FIELD_COLUMNS.items():
        key = lower.get(column.lower())
        if key is None:
            continue
        raw = row.get(key)
        text = "" if raw is None else str(raw).strip()
        if text == "" or text.lower() in ("nan", "none", "na/unk"):
            continue
        low = text.lower()
        if column in OFFICIAL_BINARY_COLUMNS:
            # 兼容三种写法：No/Yes、false/true、0/1；Margin 的 Clear=1
            if low in ("yes", "true", "1", "clear"):
                out[field] = 1
            elif low in ("no", "false", "0", "unclear"):
                out[field] = 0
            continue
        if column == "WHO_grade":
            out[field] = text.replace(".0", "")                   # Excel 常读成 3.0
            continue
        out[field] = text                                          # 枚举/多标签：原样保留
    return out


def structured_from_row(row: dict) -> dict:
    """金标准一行 → 规范字段（缺失字段不出现在结果里 → 训练时自动 mask 掉）。

    **官方英文列名优先**：命中 ``5_characteristics.xlsx`` 的列就直接返回，
    避免再过一遍中文关键词（两套命名混在一起只会互相干扰）。
    """
    official = _official_columns_to_fields(row)
    if official:
        return official

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


def build_uid_index(series_types: dict | None) -> dict[str, str]:
    """``{(检查号, 序列号): 类型}`` → ``{序列号: 类型}``（UID 单键回退索引）。

    为什么需要它：``3_serieslabel.xlsx`` 的**检查号列**与磁盘上的病例目录名并非
    总能对上（平台匿名化口径不同、前导零、目录名是哈希而表里是原始检查号），
    而 **SeriesUid 与影像同源**，是两边唯一必然一致的键。精确键查不到时按 UID
    单键回退，能把整批"看起来没模态"的病例救回来。

    在探针里**每次运行只构建一次**（表可能上万行，别放进每病例的循环里）。
    """
    out: dict[str, str] = {}
    for key, value in (series_types or {}).items():
        if not value:
            continue
        uid = key[1] if isinstance(key, (tuple, list)) and len(key) > 1 else key
        out.setdefault(_norm_key(uid), str(value))
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
        value = series_types.get((_norm_key(accession), _norm_key(uid)))
        if value:
            return str(value)
    index = uid_index if uid_index is not None else build_uid_index(series_types)
    for uid in uid_candidates:
        value = index.get(_norm_key(uid))
        if value:
            return str(value)
    return ""


def describe_modality_sources(root: str | os.PathLike | None = None) -> str:
    """一句话自检"模态来源现在什么状态"（专供报错文案，省掉一轮来回排查）。

    形如 ``标注表 3_serieslabel.xlsx=<路径或"未找到">；体素判别模型 <路径>=存在/缺失``。
    两种来源都不可用时，任何模态相关报错都会附带它 —— 用户立刻能分清是
    "表没接上"还是"体素模型没装"，不必猜。
    """
    labels = find_official_labels(root)
    series = labels.get("series")
    table_desc = (f"标注表 3_serieslabel.xlsx={series}" if series
                  else "标注表 3_serieslabel.xlsx=未找到（已搜 $GLIOMA_LABELS_DIR、"
                       "<工程>/labels、$WORKSPACE 下 3 层、数据根/父/祖父）")
    try:
        from .modality_model import DEFAULT_MODEL_PATH
        model_desc = (f"体素判别模型 {DEFAULT_MODEL_PATH}="
                      f"{'存在' if Path(str(DEFAULT_MODEL_PATH)).is_file() else '缺失'}")
    except Exception:                                     # 依赖不全也不能让报错本身再抛
        model_desc = "体素判别模型=不可用（依赖缺失）"
    return f"{table_desc}；{model_desc}"


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
    """读序列类型：``(检查号, 序列号) → 序列类型``（模态的唯一可靠来源）。

    ⚠️ 官方数据的序列目录名是 DICOM UID / 哈希，靠"按名字猜关键词"一个都命中不了：
    探针会把整批序列归到 ``other``，训练侧直接报 ``无任何可用序列`` ——
    而病例数、目录结构看起来完全正常，极易被误判成数据损坏或路径写错。

    两个来源都读，**官方优先**：

    1. **``3_serieslabel.xlsx``**（官方 ``labels/`` 下的权威来源，列 ``SeriesLabel``
       ∈ {T1CE, T2, FLAIR}）—— 见天坛参考实现 ``AIRecongition/``
    2. ``SeriesType.xlsx``（团队 README 里提到的名字，官方数据里通常并没有）

    缺文件不是错误；同一文件内同键冲突取值**直接失败**（规范 §21）。
    """
    global _WARNED_SERIES_TYPE_DEP

    out: dict[tuple[str, str], str] = {}

    # ---- ① 官方 3_serieslabel.xlsx（权威来源）----
    series_file = find_official_labels(root).get("series")
    if series_file:
        for (acc, uid), value in read_serieslabel_table(series_file).items():
            # 查表方统一用 _norm_key（去空白 + 大小写无关），这里也要归一化后再存，
            # 否则"表里大写、目录里小写"会静默查不到
            out[(_norm_key(acc), _norm_key(uid))] = value
        print(f"[labels] 已读官方序列类型 {os.path.basename(series_file)}："
              f"{len(out)} 条", flush=True)

    # ---- ② SeriesType.xlsx（兼容旧命名；不存在就跳过，**不能提前 return**）----
    path = os.path.join(str(root), "SeriesType.xlsx")
    if not os.path.isfile(path):
        return out

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
            return out

    aliases = {
        "acc": ("accessionnumber", "accession", "检查号", "检查编号", "病例号"),
        "uid": ("seriesinstanceuid", "seriesuid", "序列号", "序列uid"),
        "typ": ("seriestype", "type", "序列类型", "模态", "序列描述"),
    }
    seen_here: dict[tuple[str, str], str] = {}                     # 只用于检测本文件内的冲突
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
        if key in seen_here and seen_here[key] != value:
            raise ValueError(
                f"SeriesType.xlsx 冲突：检查号={acc!r} 序列={uid!r} "
                f"同时映射到 {seen_here[key]!r} 与 {value!r}（{path}）")
        seen_here[key] = value
        out.setdefault(key, value)                                # 官方表优先，这里只补缺
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
