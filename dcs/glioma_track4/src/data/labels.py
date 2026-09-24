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
#: 级别单独成列（`WHO分级` / `分级`）时可能出现全角数字；罗马数字见 ``_ROMAN_GRADE``
_CIRCLED_GRADE = {"Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4"}
_ROMAN_GRADE = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}

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
#:
#: ⚠️ ``accessionumber``（少一个 n）是**数据集真实表头里的拼写**，必须按精确命中列出来：
#: 只留 ``accessionnumber`` 时它进不了精确轮，而同一行的 ``StudyUid`` 能精确命中 ——
#: 于是整张检查级别 sheet 按 **StudyUid** 建索引（键形如 ``1.2.3.xxxx``），
#: 与磁盘上的检查号目录名一条都对不上，序列级/ROI级子行也全部挂不上（静默丢行）。
ID_COLUMN_KEYWORDS = ("accessionnumber", "accession_number", "accession_no",
                      "accessionumber", "accession_num", "accessionno", "accession",
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


# --------------------------------------------------------------------------- #
# 数据集金标准表的**三张工作表**（`脑胶质瘤标注结果-训练集.xlsx`）与"级别"
# --------------------------------------------------------------------------- #
#: 实测排版：三个 sheet 依次是 ``检查级别`` / ``序列级别`` / ``ROI级别``。
#:
#: ⚠️ **不要假设表头在第几行**：三个 sheet 的表头行位置各不相同，第 1~3 行
#: 都可能是"索引信息"（标题 / 字段说明 / 空行）。表头行一律由
#: :func:`_detect_header` 在**前 20 行里扫描**确定（要求含该级别的行键列），
#: 写死"第 N 行"会在官方换一版排版时整表解析出 0 行。
#:
#: 为什么必须分级别处理：三个 sheet 的**行键不是同一个** ——
#: 检查级别 = 检查号（一例一行）、序列级别 = 检查号 + 序列号（一例多行）、
#: ROI 级别 = 检查号 + 序列号 + ROI 名（一例更多行）。
#: 全部按检查号合并会让同病例的后续行**互相覆盖**：看起来"读到了 N 行"、
#: 字段却来自最后一行（序列级/ROI 级），病例级字段全丢 —— 而且不报任何错。
_SHEET_LEVEL_RULES: tuple[tuple[str, str], ...] = (
    ("检查", "case"), ("case", "case"), ("study", "case"), ("exam", "case"),
    ("序列", "series"), ("series", "series"), ("serie", "series"),
    ("roi", "roi"), ("病灶", "roi"), ("掩膜", "roi"), ("mask", "roi"),
)

#: 各级别**行键**的列名候选（合并时用它区分同一病例下的多行）
LEVEL_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "case": ID_COLUMN_KEYWORDS + ID_COLUMN_EXACT,
    "series": ("序列号", "序列编号", "序列id", "序列uid", "seriesuid", "series_uid",
               "seriesinstanceuid", "seriesid", "series"),
    # ROI 名**显式列出无下划线拼写并排最前**：官方表头是 ``RoiName``，只写 ``roi名称`` /
    # ``roi`` 时它仅被"前缀"轮的 ``roi`` 兜住 —— 而同一行还有 AB 列 ``ROIUid``（官方表的
    # 实测拼写），位置更靠前、又同样满足 ``roi`` 前缀，于是**先撞上它**：行键变成 ROI UID，
    # ROI 名（瘤体/水肿/肿瘤瘤体/全肿瘤，掩膜角色 core/peri 的唯一来源）退化成普通列。
    # 这不是理论隐患：``ROIUid`` 这种拼写靠 ``roi`` 前缀兜不住，必然被截走。
    "roi": ("roiname", "roi_name", "roi名称", "roi名", "roi编号", "roi号", "roi",
            "掩膜名", "掩膜文件", "掩膜", "maskname", "mask_name", "mask",
            "病灶名", "病灶"),
}

#: 序列级 / ROI 级的行**原样挂**在病例记录的这两个键下（不参与字段映射）。
#: 用双下划线包起来是为了与真实列名不可能撞名（列名都是中文或英文单词）。
LEVEL_NESTED_KEY: dict[str, str] = {"series": "__series_rows__", "roi": "__roi_rows__"}

#: 合并顺序：检查级别在前（病例级字段以它为准），序列/ROI 级只做嵌套保留
_LEVEL_ORDER: dict[str, int] = {"case": 0, "series": 1, "roi": 2, "unknown": 3}

#: 在**序列级 / ROI 级**表里认"检查号列"用的名字（比 :data:`ID_COLUMN_KEYWORDS` 更严）。
#:
#: 这里不能用那套宽松关键词：序列级表的行键是 ``序列号``、ROI 级是 ``ROI名称``，
#: 宽松匹配里的 ``id`` / ``编号``（包含匹配）会把 ``SeriesUid`` / ``序列编号`` 认成检查号，
#: 于是整张表的子行被挂到"序列号当检查号"的假病例上 —— 有结果、全错位。
_CASE_COLUMN_STRICT = ("accessionnumber", "accession_number", "accession_no",
                       "accessionumber", "accession_num", "accessionno", "accession",
                       "检查号", "检查编号", "病例号", "患者号", "检查序号", "检查id",
                       "studyid", "study_id", "study_instance_uid")


def _sheet_level(name: str) -> str:
    """由工作表名判"级别"：``检查级别``→case、``序列级别``→series、``ROI级别``→roi。

    名字认不出来时返回 ``"unknown"``（单张 csv、``Sheet1``、空名…）——
    **不是**"按检查号合并"：:func:`_sheet_plan` 会按表头列回退判级
    （ROI 名 → 序列号 → 检查号）。这里只负责"表名这一条线索"。
    """
    low = str(name or "").strip().casefold()
    for kw, level in _SHEET_LEVEL_RULES:
        if kw in low:
            return level
    return "unknown"


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
# 天坛参考实现 `AIRecongition/` 的 5 张表（⚠️ **不是赛道四数据集的内容**）
# --------------------------------------------------------------------------- #
#: 参考实现标注表文件名 → 用途
#:
#: ⚠️ **这 5 张表与赛道四数据集没有一点关系**（属工作区里另一个目标的产物）：
#: 赛道四的数据信息就是数据集自带的 ``SeriesType.xlsx``（见
#: :data:`SERIES_TYPE_TABLE`）与训练集的 ``脑胶质瘤标注结果-训练集.xlsx``（字段金标准）。
#: 下面这套仍保留读取，**只为兼容**：表在附近时能用上就用，读不到属正常、不是配置问题。
#:
#: | 文件 | 关键列 | 用途 |
#: |---|---|---|
#: | ``1_abnormal.xlsx`` | AccessionNumber, SeriesUid, **Label** ∈ {true,fake,compositing,duplicate} | 目标一/二的正样本（**逐序列**） |
#: | ``2_duplicate.xlsx`` | src_img, desc_img | 重复影像 pair |
#: | ~~``3_serieslabel.xlsx``~~ | ~~AccessionNumber, SeriesUid, SeriesLabel~~ | **已删除，不再读**（模态只认数据集里的 ``SeriesType.xlsx``；读它只会把 ``T2WI``/``T2-Flair`` 静默压平成 ``T2``） |
#: | ``4_masklabel.xlsx`` | AccessionNumber, SeriesUid, **Maskname** | 掩膜文件名（任意名，关键词认不出） |
#: | ``5_characteristics.xlsx`` | AccessionNumber + 14 个英文列 | 结构化字段金标准（**兼容**；数据集里那份是中文表头的 ``脑胶质瘤标注结果-训练集.xlsx``） |
#:
#: 这套约定是**唯一权威**。在此之前我们按中文列名去猜，于是出现
#: "病例数正常、一例都挑不出模态""字段金标准为空"——表一直都在，
#: 只是文件名和列名都不是我们猜的那套。
#:
#: 因此本字典里**没有** ``series`` 这个键（:data:`OFFICIAL_LABEL_FILES`）——
#: 找表、打印自检时都不会再出现 ``3_serieslabel.xlsx``。
OFFICIAL_LABEL_FILES = {
    "abnormal": "1_abnormal.xlsx",
    "duplicate": "2_duplicate.xlsx",
    "mask": "4_masklabel.xlsx",
    "characteristics": "5_characteristics.xlsx",
}

#: 序列类型表在**赛道四数据集里**的文件名（**唯一来源**）。
#:
#: 它的位置与影像同层：``<阶段>/annotation/SeriesType.xlsx``，
#: 训练/验证集的数据里都有（``training`` / ``verification``），
#: ``evaluation_*`` 评测集在正式测试时**随测试数据一起下发**。
#: 列：``AccessionNumber`` + ``SeriesUid`` + ``SeriesType``；
#: 取值共 **5 类**：``T1`` / ``T1CE（增强）`` / ``T2-Flair`` / ``T2WI`` / ``其他``。
SERIES_TYPE_TABLE = "SeriesType.xlsx"

#: 查找/自检文案里出现的表名（**只有一个名字**：数据集自带的那张）。
#: 以前这里还有 ``3_serieslabel.xlsx`` 做兜底 —— 已删除：它与本赛道数据集无关，
#: 而且取值更粗（只写 ``T2``），会把数据集里的 ``T2WI`` / ``T2-Flair`` 静默压平。
SERIES_TYPE_FILENAMES = (SERIES_TYPE_TABLE,)

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


#: 候选目录里**值得下钻一层**的子目录名（大小写无关）：标注/结果类容器。
#:
#: 为什么不无脑扫全部子目录：平台数据根下有 **3000+ 病例目录**（32 位哈希），
#: 逐个 stat 既慢，又会在备份/缓存目录里撞到同名旧表。而"标注放在哪个容器目录"
#: 其实是个有限集合 —— 平台那份在 ``training/annotation/``（英文）
#: 或下载后的 ``标注结果/``（中文），下面这些名字把两种情况都覆盖了。
#: "像标注容器"的子目录名。
#:
#: ``original`` 是**验证集**的中间层：``verification/original/`` 里既放影像也放
#: ``SeriesType.xlsx`` / ``脑胶质瘤标注结果-验证集.xlsx``。漏了它，数据根填
#: ``…/verification``（而不是 ``…/verification/original``）时表就一条都搜不到，
#: 报错只说"未找到"，看不出是差了一层目录。
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
    2. 环境变量 ``GLIOMA_LABELS_DIR``
    3. **本工程目录下的 ``labels/``**（官方默认 ``./labels``）
    4. ``$WORKSPACE`` 下名为 ``labels`` 的目录（**团队工作区里那份**；平台不随数据集下发，
       见 :func:`workspace_labels_dirs`。容器里靠这一步做到零配置）
    5. 数据根自身、父目录、祖父目录 —— 平台把 ``SeriesType.xlsx`` 放在**病例目录那一层**
       （``training/annotation/``），第 5 条就是为它准备的；传进来的若是**病例目录**，
       父/祖父两级正好覆盖到 ``annotation/``
    6. 上述每个目录下"像标注容器"的一级子目录（见 :data:`_LABEL_SUBDIR_NAMES`）——
       防止表藏在 ``annotation/`` 的下一层（下载解压后常见的 ``标注结果/``）

    抽成独立函数是因为各类表**可能不在一起**：赛道四数据集自带 ``SeriesType.xlsx``
    （与病例目录同层），而字段金标准 / 掩膜名表在**工作区**的 ``labels/`` 下；
    两套搜索必须同一口径，否则又出现"表在磁盘上、代码却只看了一个目录"。
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

    # 去重（保留优先级顺序）后，再为每个目录补上"值得下钻的子目录"（逐层展开，见
    # :data:`_LABEL_SUBDIR_DEPTH`）。用广度优先是为了保持"先浅后深"的优先级：
    # 数据根那一层永远排在自己的子目录前面。
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

    为什么不复用 :func:`find_official_labels`：它只认工作区那几张固定名字的表
    （``1_abnormal.xlsx`` 等），而赛道四数据集里那张叫 ``SeriesType.xlsx`` ——
    名字不同，必须在**同一批候选目录**里分别找，才能既认数据集又兼容工作区。
    """
    for folder in label_search_dirs(root, labels_dir):
        candidate = folder / filename
        if candidate.is_file():
            return str(candidate)
    return None


def find_official_labels(root: str | os.PathLike | None = None,
                         labels_dir: str | os.PathLike | None = None) -> dict[str, str]:
    """定位团队工作区那几张标注表 → ``{用途: 路径}``（找不到的键不出现）。

    搜索顺序见 :func:`label_search_dirs`。这些表放在**工程/工作区目录**而不是数据集里 ——
    这也是"数据根下找不到金标准"的原因之一。``root=None`` 时只搜 1~4
    （用于报错时做"表到底在不在"的自检）。

    ⚠️ 里面**没有模态表**：``3_serieslabel.xlsx`` 已从 :data:`OFFICIAL_LABEL_FILES`
    移除，模态只在数据集的 ``SeriesType.xlsx`` 里认（见 :func:`read_series_types`）。
    """
    cands = label_search_dirs(root, labels_dir)
    found: dict[str, str] = {}
    for kind, name in OFFICIAL_LABEL_FILES.items():
        for folder in cands:
            candidate = folder / name
            if candidate.is_file():
                found[kind] = str(candidate)
                break
    return found


#: 分层列名的分隔符：官方表的列名就是**字段路径**（``Study->CLINICAL->病理结果``）。
#:
#: ⚠️ 只认这两种箭头：普通连字符列名（``T2-Flair``、``t1wi_c_enhan``）不能拆。
_COLUMN_PATH_SEPS: tuple[str, ...] = ("->", "→")


def _column_leaf(name: Any) -> str:
    """取**分层列名的末段**：``Study->CLINICAL->病理结果`` → ``病理结果``。

    实测排版：检查级别 sheet 的病例级字段全写成路径形式
    （``Study->CLINICAL->病理结果``、``Study->DICOM->StudyDate``），**末段才是字段名**。

    只按整串匹配时字段能不能命中全靠"包含"，父段里出现关键词就会被抢走
    （``...->病理结果`` 与 ``...->病理类型`` 谁在前面谁得）。因此匹配顺序统一成
    **先末段、再整串** —— 只增精度，不改旧行为（非分层列名的末段就是它自己）。
    """
    text = str(name if name is not None else "").strip()
    for sep in _COLUMN_PATH_SEPS:
        if sep in text:
            text = text.rsplit(sep, 1)[-1].strip()
    return text


def _find_col_in_list(header: list[str], keywords: tuple[str, ...]) -> int | None:
    """在表头列表里找列，返回**列号**：先按**分层列名末段**、再按整串；每轮 精确→前缀→包含。

    末段优先的理由见 :func:`_column_leaf`。
    """
    full = {str(c).strip().lower(): i for i, c in enumerate(header) if str(c).strip()}
    leaf: dict[str, int] = {}
    for i, cell in enumerate(header):
        text = str(cell).strip()
        if text:
            leaf.setdefault(_column_leaf(text).lower(), i)
    for table in (leaf, full):                                     # 末段 → 整串
        for kw in keywords:                                        # ① 精确
            if kw in table:
                return table[kw]
        for kw in keywords:                                        # ② 前缀
            for low, idx in table.items():
                if low.startswith(kw):
                    return idx
        for kw in keywords:                                        # ③ 包含
            for low, idx in table.items():
                if kw in low:
                    return idx
    return None


def read_official_triples(path: str,
                          id_kws: tuple[str, ...] = ("accessionnumber", "accession", "检查号"),
                          uid_kws: tuple[str, ...] = ("seriesuid", "series_uid", "序列号"),
                          value_kws: tuple[str, ...] = ("label",),
                          ) -> dict[tuple[str, str], list[str]]:
    """读"检查号 + 序列号 + 取值"三类表 → ``{(检查号, 序列号): [取值, ...]}``。

    ``1_abnormal`` / ``4_masklabel`` 都是这个形状
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


def _sheet_frames(path: str) -> list[tuple[str, list[list[str]]]]:
    """把 csv/xlsx 读成 ``[(工作表名, 原始行), ...]``（**不做任何表头假设**）。

    为什么不直接用 ``pandas.read_excel`` 的默认行为：它把**第一行**当表头、
    且**只读第一个 sheet**。本项目两类表的真实排版都跟默认行为对不上：

    * ``脑胶质瘤标注结果-训练集.xlsx``：**3 个 sheet（检查级别/序列级别/ROI级别）**，
      每个 sheet 的表头行位置还不一样（第 1~3 行都可能是索引信息）；
    * 中文标注表的常见排版：``标题 → 空行/说明 → 真正的表头``。

    默认行为下列名会变成"标题/Unnamed"，检查号列认不出来，整表 0 行。
    工作表名要留着：它是"这张表是哪个级别"的**第一条线索**（见 :func:`_sheet_level`）；
    名字认不出来时级别由表头列决定（见 :func:`_sheet_plan`）。
    """
    if path.endswith((".xlsx", ".xls")):
        import pandas as pd
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
        return [(str(name), [["" if v is None else str(v).strip() for v in row]
                             for row in frame.fillna("").values.tolist()])
                for name, frame in sheets.items()]
    import csv
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [(os.path.splitext(os.path.basename(path))[0],
                 [[str(c).strip() for c in row] for row in csv.reader(f)])]


def _sheet_rows(path: str) -> list[list[list[str]]]:
    """只要行、不要工作表名（多数调用方用不到名字）。"""
    return [rows for _, rows in _sheet_frames(path)]


def _detect_header(rows: list[list[str]], max_scan: int = 20,
                   level: str = "unknown") -> int | None:
    """在前若干行里找**真正的表头行**：必须含该级别的行键列，字段线索越多越优先。

    只看第一行是这个数据最常见的失效点（标题行占了第一行）；
    而"必须含行键列"这条同时挡住了把说明行、数据行误判成表头。

    ``level`` 决定"行键列"是什么：检查级别看检查号，**序列级别看序列号，
    ROI 级别看 ROI 名**。少了这个参数，序列级别 / ROI 级别的 sheet 会因为
    "没有检查号列"而判成找不到表头 → 整张表 0 行（序列级字段全丢、还不报错）。
    """
    key_cols = LEVEL_KEY_COLUMNS.get(level)
    best: tuple[int, int, int] | None = None            # (字段线索, 非空列数, 行号)
    for idx, row in enumerate(rows[:max_scan]):
        cells = [str(c).strip() for c in row]
        if not any(cells):
            continue                                               # 空行
        if key_cols is not None:
            if _find_col_in_list(cells, key_cols) is None:
                continue
        elif _find_id_column(row) is None:
            continue
        hits = sum(1 for c in cells
                   if any(k in c.lower() for k in FIELD_HINTS))
        # 非空列数当**第二判据**：标题行常是"一格有字、其余全空"，而真表头是满行。
        # 少了它，标题行 'ROI级别' 会被前缀匹配当成 roi 列、压过真表头 → 表头行混进数据、
        # 还凭空多出一个用表头文字当检查号的假病例。
        score = (hits, sum(1 for c in cells if c))
        if best is None or score > best[:2]:
            best = (score[0], score[1], idx)
    return best[2] if best else None


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
    sheets = _sheet_frames(path)
    lines.append(f"工作表数：{len(sheets)}")
    for s_idx, (name, rows) in enumerate(sheets):
        width = max((len(r) for r in rows), default=0)
        level = _sheet_level(name)
        lines.append(f"\n--- sheet{s_idx} {name!r}（级别={level}）: "
                     f"{len(rows)} 行 × {width} 列 ---")
        for r_idx, row in enumerate(rows[:max_rows]):
            cells = [str(c)[:cell_width].ljust(cell_width)
                     for c in row[:max_cols]]
            more = " …" if len(row) > max_cols else ""
            lines.append(f"  行{r_idx:>3}: " + " | ".join(cells) + more)
        if len(rows) > max_rows:
            lines.append(f"  ……（还有 {len(rows) - max_rows} 行）")
    return "\n".join(lines)


def _diagnose_id_columns(sheets: list[tuple[str, list[list[str]]]], known_keys: set[str],
                         top: int = 3, max_scan: int = 300) -> str:
    """解析失败时给出**可执行的**原因：哪一列最像行键、命中多少行。

    "0 行"有两种成因，处理方式相反：
    ① 检查号列存在，但取值与磁盘目录名不是同一套编号（要按值映射或换表）；
    ② 表里根本没有检查号（这表不是字段金标准）。
    只看 `label_field_counts: {}` 分不清，只能反复猜 —— 所以把命中率打出来。

    ⚠️ 只拿**磁盘检查号**去比对每个 sheet：序列级别 / ROI 级别的 sheet
    列的是序列号 / ROI 名，命中率天然为 0，那不代表表有问题。
    """
    if not known_keys:
        return ("  ⚠️ 取不到磁盘上的检查号（数据根下没有病例目录），"
                "只能按列名识别 —— 请先把数据根指到含检查号目录的那一层。")
    lines: list[str] = []
    for s_idx, (sheet_name, rows) in enumerate(sheets):
        tag = f"sheet{s_idx} {sheet_name!r}"
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
            lines.append(f"  {tag}: 没有哪一列的取值能对上磁盘检查号"
                         f"（前几列表头={head}）→ 该 sheet 多半不是订单级别、或不是这张表")
            continue
        for ratio, hit, total, col in scored[:top]:
            name = ""
            for r in rows[:20]:                                        # 该列的表头文字
                if col < len(r) and str(r[col]).strip() and _id_key(r[col]) not in known_keys:
                    name = str(r[col]).strip()[:18]
                    break
            lines.append(f"  {tag}: 第 {col} 列最像检查号（表头 {name!r}）"
                         f"命中 {hit}/{total} 行（{ratio:.0%}）")
        lines.append(f"  → 上面命中率若明显低于 30%，说明该表编号与目录名不是同一套；"
                     f"把 dump 出来的前几行发出来即可确定映射关系")
    return "\n".join(lines)


def _row_to_record(header: list[str], row: list[str]) -> dict[str, Any]:
    """一行 + 表头 → ``{列名: 值}``（表头认不出时用 ``colN`` 占位）。"""
    record: dict[str, Any] = {
        col: (str(row[i]).strip() if i < len(row) else "")
        for i, col in enumerate(header) if col
    }
    if not record:                                                # 表头认不出：用列号占位
        record = {f"col{i}": (str(row[i]).strip() if i < len(row) else "")
                  for i in range(len(row))}
    return record


def _key_forms(value: str) -> set[str]:
    """登记/查找用的键形式：原样、去前导零、以及两者的 casefold，再加归一化键。

    目录名可能是 ``C0E1F8F2-53BA-45BE``，表里是 ``c0e1f8f253ba45be``（或反之），
    只登记原始形式会一条都查不到。

    ⚠️ 归一化后是**空串**的键一律丢掉：``_id_key("检查号")`` 这种"整串都是非字母数字"
    的取值会归一化成 ``""``，留着就会把不同来源的无意义文字**合并成同一个病例**。
    """
    text = str(value).strip()
    stripped = text.lstrip("0") or text
    return {f for f in (text, stripped, text.casefold(), stripped.casefold(),
                        _id_key(text)) if f}


def _case_record(out: dict[str, dict], case_id: str, create: bool = True) -> dict | None:
    """取（必要时新建）某检查号的病例记录，并把**它的所有键形式都指到同一个对象**。

    序列级 / ROI 级的行要挂到病例上，而病例记录可能还没被创建
    （表里只有序列级 / ROI 级 sheet），也可能已由检查级别 sheet 建好 ——
    两条路径必须落到**同一个 dict**，否则后挂的子行会凭空消失。

    ``create=False`` 时只认**已存在**的病例（找不到返回 ``None``）：用于
    "检查号列是靠列名猜出来的"这种不够可靠的场景，避免把 ``SeriesId`` 之类的
    取值当成检查号、凭空造出一批假病例。
    """
    forms = _key_forms(case_id)
    record = next((out[k] for k in forms if k in out), None)
    if record is None:
        if not create:
            return None
        record = {}
    for key in forms:
        out[key] = record
    return record


def _fill_gaps(base: dict, record: dict) -> None:
    """把 ``record`` 里**非空、且 base 还没有**的列填进 base（先到的值优先）。

    多个 sheet 都可能带病例级字段时用它合并：后一张表只补空，
    不会用空值 / 粗粒度取值把前面（权威）的表覆盖掉。
    """
    for col, value in record.items():
        if isinstance(value, list):                                # 子行列表：合并而非覆盖
            base.setdefault(col, []).extend(value)
            continue
        if value == "" or base.get(col, "") != "":
            continue
        base[col] = value


#: 允许从序列级 / ROI 级子行"提"到病例级的**标签列**（三组：各取一个；见 :func:`_promote_case_labels`）。
#:
#: 只认这三组，是因为**只有丢它们才会静默缩小评测分母**（``TumorProbability`` / ``WHO_Grade``
#: 直接从这三组来）。其余字段本来就该留在各自级别上，见 :func:`_promote_case_labels`。
_CASE_LABEL_COLUMN_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("病理结果", "pathology", "病理"), ()),
    (("WHO分级", "WHO_grade", "WHO grade", "分级"), ("病理", "pathology")),
    (("glioma_with_label", "胶质瘤"), ()),
)


def _promote_case_labels(out: dict[str, dict]) -> None:
    """把序列级 / ROI 级子行里**取值一致**的**标签列**补进病例级（只补，不覆盖已有值）。

    为什么需要：病例**只出现在**序列级 / ROI 级 sheet 里时（``检查级别`` sheet 没这一行），
    病例记录是**空的** —— 而官方表同样把 ``Study->CLINICAL->病理结果`` 放在
    **ROI 级别 sheet 的 AQ 列**。整条丢掉的表现是"这一例没有金标准"：
    评测分母悄悄变小、还不报错。

    为什么**只提标签列**、而不是把整行兜进去（试过，是错的）：
        序列级 / ROI 级是"**一例多行**"，除标签外的列本来就**逐行不同**。实测复现数据里
        同一病例的两条序列行 ``Signal_T2WI`` 分别是 ``High`` / ``Low`` —— 整行兜底会
        **任取一行**当金标准，表现是"读到了值、而且是错的"，比读不到更难查。
        所以这里：

    * 只认**标签列**（病理结果 / WHO 分级 / Glioma）；
    * 该列在本病例**所有**子行里的非空取值必须完全一致，不一致就整体不采纳并告警；
    * ``检查级别`` sheet 已给非空值的（同末段列名）一律不动 —— 它才是权威（见 README §2.2）。
    """
    done: set[int] = set()
    for rec in out.values():
        if id(rec) in done:                                        # 同一记录有多个别名键
            continue
        done.add(id(rec))
        rows = [r for key in LEVEL_NESTED_KEY.values() for r in (rec.get(key) or [])]
        if not rows:
            continue
        for keywords, exclude in _CASE_LABEL_COLUMN_GROUPS:
            values: set[str] = set()
            col_name = ""
            for row in rows:
                col = _find_col(row, list(keywords),
                                list(exclude) if exclude else None)
                if not col:
                    continue
                col_name = col_name or col
                text = str(row.get(col) if row.get(col) is not None else "").strip()
                if text:
                    values.add(text)
            if not values:
                continue
            if len(values) > 1:                                    # 子行自相矛盾：不猜
                print(f"[labels][告警] 子行里 {col_name!r} 取值不一致 "
                      f"{sorted(values)}，不提升为病例级（避免任取一行当金标准）",
                      flush=True)
                continue
            leaf = _column_leaf(col_name).casefold()
            same_leaf = [k for k in rec
                         if not _is_nested_key(k)
                         and _column_leaf(k).casefold() == leaf]
            if any(str(rec.get(k) or "").strip() for k in same_leaf):
                continue                                           # 检查级别已给：权威
            value = next(iter(values))
            if same_leaf:
                rec[same_leaf[0]] = value                          # 填掉同字段列的空格
            else:
                rec[col_name] = value


def _sheet_plan(rows: list[list[str]], level: str, known_ids: set[str] | None,
                ) -> tuple[list[str], int, int, str] | None:
    """定出这张表的 ``(表头, 数据起始行, 行键列, 实际级别)``；认不出来返回 ``None``。

    分两套走，因为各级别 sheet 的**行键不同**：

    * ``case``：行键是**检查号** —— 先按取值比对磁盘目录名
      （不看列名，最可靠），不行再退回按列名认；
    * ``series`` / ``roi``：行键是**序列号 / ROI 名** —— 只能按列名认。
      这两级 sheet 里没有检查号行键，若还按老逻辑"必须含检查号列"，
      整张表会判成"找不到表头"→ 0 行。

    返回的级别可能与入参 ``level`` 不同：工作表名认不出来（``unknown``）时，
    这里会**按行键列名回退判级**（见下方 ``_FALLBACK_LEVELS``）。
    调用方必须用返回的这个级别去决定"这行怎么挂" —— 工作表名只是线索，
    表头列才是事实。
    """
    if level == "unknown":
        # 工作表名认不出级别（``Sheet1`` / ``序列信息`` / ``病灶`` / 空名…）时，
        # 按"这张表有哪种行键列"判级，顺序**由具体到宽泛**：
        # ROI 级最具体（有 ROI 名）→ 序列级（有序列号）→ 检查级（只有检查号）。
        #
        # 顺序反了的代价：序列级 / ROI 级 sheet 里**同样有检查号列**（要把子行挂到
        # 病例上），若先按检查号判成 case，同病例的多行会互相覆盖 ——
        # 字段看着有值、实际只留最后一行，且不报任何错。
        for cand in _FALLBACK_LEVELS:
            found = _sheet_plan(rows, cand, known_ids)
            if found is not None:
                return found[0], found[1], found[2], cand
        return None

    if level == "case":
        if known_ids:                                              # ① 按取值找检查号列
            found = _best_id_column_by_values(rows, known_ids)
            if found is not None:
                id_col, first_data = found
                head = _header_row_above(rows, first_data)
                if head is not None:
                    return ([str(c).strip() for c in rows[head]], first_data, id_col,
                            "case")
        head = _detect_header(rows, level="case")                   # ② 按列名找
        if head is None:
            return None
        header = [str(c).strip() for c in rows[head]]
        id_name = _find_id_column(header)
        if not id_name:
            return None
        return header, head + 1, header.index(id_name), "case"

    key_cols = LEVEL_KEY_COLUMNS[level]
    head = _detect_header(rows, level=level)
    if head is None:
        return None
    header = [str(c).strip() for c in rows[head]]
    key_col = _find_col_in_list(header, key_cols)
    if key_col is None:
        return None
    return header, head + 1, key_col, level


#: 工作表名判不出级别时的回退判级顺序（具体 → 宽泛，理由见 :func:`_sheet_plan`）。
_FALLBACK_LEVELS: tuple[str, ...] = ("roi", "series", "case")


def _find_case_column(rows: list[list[str]], header: list[str], key_col: int,
                      known_ids: set[str] | None) -> tuple[int | None, bool]:
    """在**序列级 / ROI 级**表里找"检查号列" → ``(列号, 是否可信)``。

    找不到返回 ``(None, False)``，那批子行就只能是"挂不上病例"。

    为什么两套、还带回一个"可信"标志：

    * **按取值**（可信）：拿磁盘上的检查号去比对，不看列名 —— 命中了就是真检查号列，
      用它建病例记录是安全的；
    * **按列名**（不可信）：离线看表（没有 ``known_ids``）时的兜底，用的是
      :data:`_CASE_COLUMN_STRICT` 这套**更严**的名字。既然只是猜的，就
      **只挂到已存在的病例上**（``create=False``）—— 猜错时最多丢几行，
      而不会凭 ``SeriesId`` 造出一批字段全空的假病例。

    无论哪套都必须排除 ``key_col``：序列级表的 ``序列号`` 满足宽松关键词里的
    ``id`` / ``编号``（包含匹配），一不小心就把行键当检查号。
    """
    if known_ids:
        found = _best_id_column_by_values(rows, known_ids)
        if found is not None and found[0] != key_col:
            return found[0], True
    col = _find_col_in_list(header, _CASE_COLUMN_STRICT)
    if col is not None and col != key_col:
        return col, False
    return None, False


def read_structured_table(path: str, known_ids: set[str] | None = None) -> dict[str, dict]:
    """读结构化金标准表（csv 或 xlsx）→ ``{检查号: {列名: 值}}``。

    同一文件里可能**按级别分成多张工作表**（``脑胶质瘤标注结果-训练集.xlsx``
    就是 ``检查级别`` / ``序列级别`` / ``ROI级别`` 三张），四个坑各自有对策：

    1. **表头行不固定**：排版是 ``标题行/索引信息 → 空行/说明 → 真正的表头``，
       每个 sheet 还不一样（第 1 行、第 2 行、第 3 行都可能是索引信息），
       而 ``pandas.read_excel`` 默认把第一行当表头 —— 列名成了"标题/Unnamed"、
       检查号列认不出来，整表解析出 **0 行**。这里改为**扫描前若干行**找表头
       （见 :func:`_detect_header`：必须含该级别的行键列，字段线索最多者胜），
       不假设它在第几行。
    2. **多工作表 + 级别判定**：默认只读第一个 sheet；这里遍历全部 sheet。
       级别先按**工作表名**判（见 :func:`_sheet_level`），名字认不出来
       （``Sheet1`` / ``序列信息`` / 空名）再按**表头列**回退判级
       （ROI 名 → 序列号 → 检查号，见 :func:`_sheet_plan` 的 ``_FALLBACK_LEVELS``）。
       只看工作表名是不够的：认不出时一张序列级 sheet 会被当成检查级别 ——
       同病例的多行互相覆盖，字段看着有值、实际只留最后一行，且不报错。
    3. **行键不同**：检查级别一例一行（键=检查号）、序列级别一例多行
       （键=检查号+序列号）、ROI 级别更多行（键=+ROI 名）。全按检查号合并会让
       同病例的后续行**互相覆盖** —— 字段看着有值，实际来自最后一行（序列级），
       病例级字段全丢，且不报任何错误。所以序列级 / ROI 级的行**原样挂**在病例
       记录的 ``__series_rows__`` / ``__roi_rows__`` 下，不参与字段映射。
    4. **检查号大小写**：目录名是小写哈希、表里可能是大写，
       因此大小写折叠后的键也一并登记。

    一行都没解析出来时会打印告警（含表头预览 + 整表摊开），
    而不是只给上层返回一个空字典。
    """
    out: dict[str, dict] = {}
    sheets = _sheet_frames(path)
    # 先给每张工作表"定级别 + 定表头"，再按**实际级别**排序：
    # 检查级别先合并（病例级字段以它为准），序列 / ROI 级只做嵌套保留。
    #
    # 为什么排序也用实际级别：工作表名可能认不出来（``Sheet1`` / ``序列信息`` /
    # 甚至空名），此时级别由表头列决定（见 :func:`_sheet_plan` 的回退判级）。
    # 若排序仍按表名，认不出名字的**检查级别** sheet 会被排到序列级之后，
    # 它的病例级字段只能"补空"（``_fill_gaps``）——权威取值被前一张表压住。
    prepared: list[tuple[int, str, list[list[str]], str, tuple | None]] = []
    for idx, (name, rows) in enumerate(sheets):
        named = _sheet_level(name)
        prepared.append((idx, name, rows, named,
                         _sheet_plan(rows, named, known_ids) if rows else None))
    ordered = sorted(prepared,
                     key=lambda p: (_LEVEL_ORDER[p[4][3] if p[4] else p[3]], p[0]))
    stats: list[str] = []
    for _, sheet_name, rows, named_level, plan in ordered:
        if not rows:
            continue
        label = sheet_name or f"sheet{len(stats)}"
        if plan is None:
            extra = ("（表名认不出级别，已按 ROI→序列→检查 顺序试过行键列）"
                     if named_level == "unknown" else "")
            stats.append(f"{label}={named_level}/表头未识别{extra}")
            continue
        header, data_start, key_col, level = plan
        # 表名与表头列不一致时以表头列为准，并在统计里标出来（否则会让人以为走错了分支）
        shown = level if level == named_level else f"{level}（表名判为 {named_level}）"
        nested_key = LEVEL_NESTED_KEY.get(level)
        # 序列级 / ROI 级：还得分清"哪一列是检查号"（好把子行挂到病例上）。
        # 检查号列不可信时只挂已有病例（create=False），见 _find_case_column。
        case_col: int | None = None
        case_trusted = False
        if nested_key is not None:
            case_col, case_trusted = _find_case_column(rows, header, key_col, known_ids)

        n_rows = n_dropped = 0
        for row in rows[data_start:]:
            if not any(str(c).strip() for c in row):
                continue                                          # 跳过空行
            key_value = str(row[key_col]).strip() if key_col < len(row) else ""
            if not key_value:
                continue
            record = _row_to_record(header, row)
            if nested_key is None:
                forms = _key_forms(key_value)
                if not forms:                                     # 取值归一化后空：无意义行
                    n_dropped += 1
                    continue
                for key in forms:
                    if key not in out:
                        out[key] = record
                    else:
                        _fill_gaps(out[key], record)
            else:
                attach = (str(row[case_col]).strip()
                          if case_col is not None and case_col < len(row) else "")
                target = (_case_record(out, attach, create=case_trusted)
                          if attach else None)
                if target is None:                                # 挂不上病例：不进结果
                    n_dropped += 1
                    continue
                target.setdefault(nested_key, []).append(record)
            n_rows += 1
        stats.append(f"{label}={shown}/{n_rows} 行"
                     + (f"（{n_dropped} 行挂不到病例）" if n_dropped else ""))

    _promote_case_labels(out)

    if stats:
        print(f"[labels] {os.path.basename(path)} 分级解析：" + "，".join(stats)
              + f" → {len({id(v) for v in out.values()})} 例", flush=True)

    if not out and path not in _WARNED_TABLES:
        _WARNED_TABLES.add(path)
        diag = _diagnose_id_columns(sheets, {_id_key(k) for k in (known_ids or ())})
        print(f"[labels][告警] {os.path.basename(path)} 未解析出任何行。\n"
              f"  已尝试：按工作表名分级 → 按取值比对检查号列 → 按列名识别行键 → 跳过标题行。\n"
              + (diag + "\n" if diag else "")
              + f"  下面把表整个摊开，直接看它长什么样：\n{dump_table(path)}", flush=True)
    return out


def _is_nested_key(key: Any) -> bool:
    """是不是"序列级 / ROI 级子行"挂载用的保留键（``__series_rows__`` 之类）。

    它们**不是真实列**，字段映射必须跳过：否则子行列表会被当成字段取值，
    或反过来被 ``_find_col`` 的包含匹配命中（``__series_rows__`` 里就含 "series"）。
    """
    return str(key).startswith("__")


def _find_col(row: dict, keywords: list[str], exclude: list[str] | None = None) -> str | None:
    """列名匹配：**先按分层列名末段、再按整串**；每轮内部 精确 → 前缀 → 包含。

    ``exclude`` 用于排除干扰列（如 ``*_pattern``）。早期版本直接"包含匹配"会让
    ``t1wi_c_enhan`` 命中 ``tumor_t1wi_c_enhan_pattern``，Enhancement 永远拿不到金标准。

    末段优先：官方表的列名是字段路径（``Study->CLINICAL->病理结果``），末段才是字段名；
    整串匹配会让父段里的关键词抢列（见 :func:`_column_leaf`）。
    """
    ex = [e.lower() for e in (exclude or [])]
    items = [(str(k).strip(), k) for k in row if not _is_nested_key(k)]
    full: dict[str, Any] = {text.lower(): key for text, key in items}
    leaf: dict[str, Any] = {}
    for text, key in items:
        leaf.setdefault(_column_leaf(text).lower(), key)
    for table in (leaf, full):                                     # 末段 → 整串
        for kw in keywords:                                        # 1) 精确
            if kw.lower() in table:
                return table[kw.lower()]
        for kw in keywords:                                        # 2) 前缀
            for low, key in table.items():
                if low.startswith(kw.lower()) and not any(e in low for e in ex):
                    return key
        for kw in keywords:                                        # 3) 包含
            for low, key in table.items():
                if kw.lower() in low and not any(e in low for e in ex):
                    return key
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
    lower = {str(k).strip().lower(): k for k in row if not _is_nested_key(k)}
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


def _grade_from_text(text: Any) -> str | None:
    """从任意写法里抠 WHO 级别 → ``"1".."4"``（认不出返回 ``None``）。

    同一件事在表里有四种写法：``4`` / ``4级`` / ``Ⅳ`` / ``IV``。只做精确匹配
    （:data:`GRADE_MAP` 的 ``脑胶质瘤N级``）时，``胶质瘤WHO 4级`` 这类写法整条丢掉 ——
    字段缺失会被静默当成"这一例没有金标准"，指标分母悄悄变小，看起来一切正常。
    """
    t = str(text if text is not None else "").strip()
    if not t:
        return None
    m = re.search(r"([1-4])\s*级", t)                              # `4级` / `WHO 4 级`
    if m:
        return m.group(1)
    m = re.search(r"([ⅠⅡⅢⅣ])", t)                                 # 全角 `Ⅳ级`
    if m:
        return _CIRCLED_GRADE[m.group(1)]
    t2 = t.replace(".0", "").strip()                               # Excel 常读成 `4.0`
    if t2 in _ROMAN_GRADE.values():
        return t2
    m = re.search(r"\b(iv|iii|ii|i)\b", t, re.IGNORECASE)          # 罗马数字 `IV`
    return _ROMAN_GRADE[m.group(1).lower()] if m else None


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
        elif "胶质瘤" in patho:
            # 更松的写法：`胶质瘤WHO 4级` / `胶质瘤Ⅳ级` / 只写 `胶质瘤`（级别在另一列）。
            # 先认病名（含"胶质瘤"就是阳性），级别能从文字里抠出来就顺手用上。
            out["TumorProbability"] = 1
            grade = _grade_from_text(patho)
            if grade:
                out["WHO_Grade"] = grade
        elif patho in NON_GLIOMA or any(k in patho for k in ("转移", "脓肿", "梗死", "无")):
            out["TumorProbability"] = 0

    if "WHO_Grade" not in out:
        # 级别单独占一列的表（`WHO分级` / `分级`，取值 1~4 或 I~IV）
        grade_col = _find_col(row, ["WHO分级", "WHO_grade", "WHO grade", "分级"],
                              exclude=["病理", "pathology"])
        grade = _grade_from_text(row.get(grade_col)) if grade_col else None
        if grade:
            out["WHO_Grade"] = grade
            out.setdefault("TumorProbability", 1)

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

    为什么需要它：类型表的**检查号列**与磁盘上的病例目录名并非总能对上
    （平台匿名化口径不同、前导零、目录名是哈希而表里是原始检查号），
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

    形如 ``数据信息表 SeriesType.xlsx=<路径或"未找到">；体素判别模型 <路径>=存在/缺失``。
    两者都不可用时，任何模态相关报错都会附带它 —— 用户立刻能分清是"表没接上"
    还是"体素模型没装"，不必猜。

    ``root`` 传**病例目录**也行：候选目录含数据根/父/祖父，正好覆盖到
    ``training/annotation/``（``SeriesType.xlsx`` 与病例目录同层）。
    """
    found = [(name, find_named_table(name, root)) for name in SERIES_TYPE_FILENAMES]
    table_desc = "；".join(f"数据信息表 {name}={path}" for name, path in found if path)
    if not table_desc:
        table_desc = (f"数据信息表 {SERIES_TYPE_TABLE}=未找到（已搜 "
                      "$GLIOMA_LABELS_DIR、<工程>/labels、$WORKSPACE 下 3 层、数据根/父/祖父"
                      "+ 像标注容器的子目录；它就在数据里、与病例目录同层）")
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


#: 类型表的列名候选（顺序即优先级；用**包含**匹配，故短词靠后）。
_SERIES_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "acc": ("accessionnumber", "accession", "检查号", "检查编号", "病例号"),
    "uid": ("seriesinstanceuid", "seriesuid", "序列号", "序列uid"),
    "typ": ("seriestype", "type", "序列类型", "模态", "序列描述"),
}

#: "像模态取值"的前缀（用于**按取值**找类型列，见 :func:`_sniff_series_type_columns`）。
#: 配上长度上限后，检查号 / 序列号这类长串一律不会命中。
_MODALITY_VALUE_TOKENS = (
    "t1", "t1c", "t1ce", "t1wi", "t1w", "t2", "t2w", "t2wi", "t2flair", "flair",
    "其他", "其它", "other", "none", "无", "增强", "平扫", "adc", "dwi",
)
#: 模态取值的长度上限：``T1CE（增强）`` 也就 8 个字符，长串必不是模态。
_MODALITY_VALUE_MAXLEN = 16


def _looks_like_modality_value(value) -> bool:
    """该单元格"看着像模态取值"吗（专供列名认不出时的取值嗅探）。"""
    text = re.sub(r"[\s\-_/]+", "", str(value if value is not None else "")).casefold()
    if not text or len(text) > _MODALITY_VALUE_MAXLEN:
        return False
    return any(text == tok or text.startswith(tok) for tok in _MODALITY_VALUE_TOKENS)


def _looks_like_series_type_header(acc, uid, typ) -> bool:
    """这三个取值是不是"类型表的表头行"。

    多张工作表拼成一个 ``rows`` 后，**每张表都带一遍表头**；不跳过就会出现
    ``(accessionnumber, seriesuid) → SeriesType`` 这种拿表头文字当数据的脏条目。
    """
    def _hit(value, keys: tuple[str, ...]) -> bool:
        text = _norm_key(value)
        return bool(text) and any(k in text for k in keys)

    return (_hit(acc, _SERIES_TYPE_ALIASES["acc"])
            and _hit(uid, _SERIES_TYPE_ALIASES["uid"])
            and _hit(typ, _SERIES_TYPE_ALIASES["typ"]))


def _sniff_series_type_columns(rows: list[list], max_scan: int = 300
                               ) -> tuple[int, int, int] | None:
    """**不看列名**，按取值找出 ``(检查号列, 序列号列, 类型列)``；认不出返回 ``None``。

    为什么需要它：列名是唯一会被"改版"的东西 —— 前两列写成 ``序号/影像编号``、
    加了索引列、或者干脆是 ``A/B/C``，按列名匹配就一条也读不到，
    而**取值**不会变（检查号、DICOM UID、5 类模态取值）。

    判据（全在取值上，不需要任何外部信息）：

    * **类型列**：该列非空取值里"像模态取值"的比例最高且 ≥ 0.5；
    * **序列号列**：剩下两列里，取值含 ``.``（DICOM UID）比例更高 /
      平均更长的那个；
    * **检查号列**：另一个。

    找不到（例如整表只有两列、或该列取值是自由文本）就返回 ``None``，
    绝不在没有把握时硬凑 —— 凑错会把整表挂到错误的键上，比读不到更难查。
    """
    body = [r for r in rows[:max_scan] if any(str(c).strip() for c in r)]
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


def read_series_types(root: str | os.PathLike,
                      labels_dir: str | os.PathLike | None = None
                      ) -> dict[tuple[str, str], str]:
    """读序列类型：``(检查号, 序列号) → 序列类型``（模态的唯一可靠来源）。

    ⚠️ 数据里的序列目录名是 DICOM UID / 哈希，靠"按名字猜关键词"一个都命中不了：
    探针会把整批序列归到 ``other``，训练侧直接报 ``无任何可用序列`` ——
    而病例数、目录结构看起来完全正常，极易被误判成数据损坏或路径写错。

    **只读数据集自带的 ``SeriesType.xlsx``** —— 与病例目录**同层**
    （训练集 ``<阶段>/annotation/``、验证集 ``<阶段>/original/``）；
    ``training`` / ``verification`` 的数据里都有，``evaluation_*`` 评测期
    **随测试数据一起下发**。列 ``AccessionNumber / SeriesUid / SeriesType``，
    取值 5 类：``T1`` / ``T1CE（增强）`` / ``T2-Flair`` / ``T2WI`` / ``其他``。
    它不只在数据根那一层 —— 数据根指成**某一病例目录**或**填高一层**时也要能找到，
    所以统一按候选目录搜（数据根/父/祖父 + 像标注容器的子目录，含 ``original/``）。

    读取上**不假设任何排版**：表头行是哪一行由"能否凑齐三列名"扫出来
    （实测第 1 行，带标题/索引行的版本同样能认）；列名一条都不命中时
    还会按**取值**嗅探三列（见 :func:`_sniff_series_type_columns`），
    所以列名改版、前两列是索引列都读得到。

    **工作区那份 ``3_serieslabel.xlsx`` 一律不读**：它不是本赛道数据集的内容，
    且取值更粗（只有 ``T1CE``/``T2``/``FLAIR``），一旦参与合并就会把 ``T2WI`` /
    ``T2-Flair`` 静默压平、把 ``其他`` 变成假 ``FLAIR`` ——
    表现是"模态看着都认出来了、通道里却是错的对比度"，比直接报错难查得多。

    缺文件不是错误（返回空表，由上层报错时附自检）；同一文件内同键冲突取值
    **直接失败**（规范 §21）。
    """
    global _WARNED_SERIES_TYPE_DEP

    out: dict[tuple[str, str], str] = {}

    path = find_named_table(SERIES_TYPE_TABLE, root, labels_dir)
    if not path:
        return out                                                # 表没接上：交给上层自检

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

    aliases = _SERIES_TYPE_ALIASES
    # ① 先按**列名**找表头行：哪一行能凑齐三列就用哪一行，不假设第几行
    #    （实测表头就在第 1 行；标题/索引行占位的版本也照样能认出来）。
    idx: dict[str, int] = {}
    data_start = 0
    for i, row in enumerate(rows):
        header = [_norm_key(c) for c in row]
        if not any(header):
            continue
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
        # ② 列名一条都没命中（改版 / 前两列是索引 / 英文缩写）→ 按**取值**嗅探三列。
        #    这是唯一不依赖列名的手段：检查号、DICOM UID、5 类模态取值本身就有形态，
        #    而"列名"是唯一会被改版改掉的东西（含空格、全角括号、加后缀…）。
        sniff = _sniff_series_type_columns(rows)
        if sniff is None:
            if not _WARNED_SERIES_TYPE_DEP:
                _WARNED_SERIES_TYPE_DEP = True
                print(f"[probe][告警] {os.path.basename(path)} 里既没找到 "
                      f"AccessionNumber/SeriesUid/SeriesType 三列、也没能按取值嗅探出它们"
                      f"（{path}）→ 序列类型读不到。请把表头行原样贴出来。", flush=True)
            return out
        acc_col, uid_col, typ_col = sniff
        idx = {"acc": acc_col, "uid": uid_col, "typ": typ_col}
        data_start = 0
        sniffed = True
        print(f"[labels][告警] {os.path.basename(path)} 的列名未识别 → 已按取值定位："
              f"检查号=第 {acc_col + 1} 列、序列号=第 {uid_col + 1} 列、"
              f"类型=第 {typ_col + 1} 列（读到 {len(rows)} 行）。"
              f"若取值明显不对，把表的前几行贴出来。", flush=True)

    seen_here: dict[tuple[str, str], str] = {}                     # 只用于检测本文件内的冲突
    for row in rows[data_start:]:
        try:
            acc, uid, typ = row[idx["acc"]], row[idx["uid"]], row[idx["typ"]]
        except IndexError:
            continue
        if acc in (None, "") or uid in (None, "") or typ in (None, ""):
            continue
        if sniffed:
            # 嗅探模式下靠取值过滤：表头行、说明行的"类型"取值不像模态
            if not _looks_like_modality_value(typ):
                continue
        elif _looks_like_series_type_header(acc, uid, typ):
            continue                                              # 多 sheet 拼接出的重复表头
        key = (_norm_key(acc), _norm_key(uid))
        value = str(typ).strip()
        if not value:
            continue
        if key in seen_here and seen_here[key] != value:
            raise ValueError(
                f"SeriesType.xlsx 冲突：检查号={acc!r} 序列={uid!r} "
                f"同时映射到 {seen_here[key]!r} 与 {value!r}（{path}）")
        seen_here[key] = value
        out[key] = value
    if seen_here:
        print(f"[labels] 已读序列类型表 {os.path.basename(path)}：{len(seen_here)} 条"
              f"（{path}）", flush=True)
    return out


#: 类型表里**明确表示"不是目标模态"**的取值。
#:
#: 平台下发的 ``SeriesType.xlsx`` 给**每一路**序列都标了类型，不属于
#: T1 / T2-Flair / T1CE（增强）的写成 ``其他``。这是**权威结论**，不能再当成
#: "没认出来"丢给体素模型猜：模型只认识 t1c/t2/flair/t1 四类，把一路 DWI/ADC
#: 判成 ``t2``（置信度往往还不低）会往通道里灌错对比度 —— 比留一个空通道更有害；
#: 顺带还白跑一次模型、也把 ``unknown_series_total`` 这个诊断指标带偏。
EXPLICIT_OTHER_VALUES = frozenset({
    "其他", "其它", "其他序列", "非目标", "other", "others", "none", "na", "n/a", "无",
})


def is_explicit_other(text: Any) -> bool:
    """该序列类型是否**明确写着"其他"**（而不是"没写/没认出来"）。

    只认**全等**，不做包含匹配：``其他肿瘤或病变`` 是病灶/病理取值（见
    :data:`NON_GLIOMA`），不是"序列类型=其他"，混在一起会把真正的影像丢掉。
    """
    return _norm_key(text) in EXPLICIT_OTHER_VALUES


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


#: 非"字段金标准表"的文件名关键词（见 :func:`find_structured_tables`）。
#: 两个序列类型表的命名（数据集里的 ``SeriesType`` / 工作区那份 ``3_serieslabel``）
#: 都在其中：它们的列是 ``AccessionNumber/SeriesUid/SeriesType|SeriesLabel``，
#: 一行字段都映射不出来，混进来只会让"解析出 N 行 / 有表但没解析出"这两句诊断互相矛盾。
#: （后者虽已不读，但仍要在**找表**时排除 —— 否则诊断计数还是会被它带偏。）
_NON_LABEL_TABLE_KW = ("seriestype", "series_type",          # 模态表（数据集自带）
                       "serieslabel", "series_label",        # 模态表（工作区那份，已不读）
                       "masklabel", "mask_label",            # 掩膜名表（4_masklabel.xlsx）
                       "gold", "duplicate", "folds")


def find_structured_tables(root: str, max_parents: int = 2) -> list[str]:
    """在数据根（及其**上级 1~2 层**）找结构化金标准表（csv/xlsx）。

    为什么要向上看：官方数据的层级通常是 ``<数据集根>/<某层>/<检查号>/``，
    而字段金标准表常常放在**检查号那一层的上一级**。只扫数据根会出现
    "表明明存在、却一条都没读进来"，报告里只显示 ``{}``，
    分不清是"没表"还是"数据根定位偏了一层"。

    排除另有专用读取器的表（见 :data:`_NON_LABEL_TABLE_KW`）：序列类型表
    （``SeriesType.xlsx`` / ``3_serieslabel.xlsx``）、掩膜名表（``4_masklabel.xlsx``）、
    重复影像金标准（``gold*.csv`` / ``2_duplicate.xlsx``）—— 它们都不是字段金标准，
    混进来会污染"解析出 N 行"这个计数，把诊断信息带偏。

    唯一**保留**的官方表是 ``5_characteristics.xlsx``（14 个英文列的字段金标准）。
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
