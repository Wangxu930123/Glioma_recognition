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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.selector import pick_series
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
# 病例发现
# --------------------------------------------------------------------------- #
def discover_cases(dataset_root: Path, limit: int | None = None) -> list[dict]:
    """扫描 ``<root>/<AccessionNumber>/<SeriesUid>/*.nii[.gz]``。

    返回 ``[{"accession", "dir", "series": [{"path","uid"}]}, ...]``。
    只做轻量发现（不读体素），真正的读取延迟到 ``load_case``。
    """
    root = Path(dataset_root)
    assert_case_root(root)
    cases: list[dict] = []
    for acc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if acc_dir.name.lower() in ("annotation", "cache", "runs"):
            continue
        series: list[dict] = []
        for f in sorted(acc_dir.rglob("*")):
            if not f.is_file():
                continue
            name = f.name.lower()
            if not name.endswith(NIFTI_SUFFIXES):
                continue
            if any(h in name for h in MASK_HINTS):
                continue
            series.append({"path": str(f), "uid": f.parent.name})
        if series:
            cases.append({"accession": acc_dir.name, "dir": str(acc_dir), "series": series})
        if limit and len(cases) >= limit:
            break
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
        print(f"[data] {p} 的 fold={fold} 与当前数据不匹配"
              f"（train={len(trc)} val={len(vac)}），改用 val_ratio 划分", flush=True)
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
    """
    out: dict[str, list[str]] = {}
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
        is_flair_like = any(k in parent for k in ("flair", "t2"))
        core_like = any(k in name for k in ("core", "核心", "瘤体", "增强", "et", "肿瘤"))
        peri_like = any(k in name for k in ("flair", "peri", "水肿", "异常", "whole", "总"))
        if peri_like and not core_like:
            out.setdefault("peri", []).append(str(f))
        elif core_like:
            out.setdefault("peri" if is_flair_like else "core", []).append(str(f))
        elif peri_like:
            out.setdefault("peri", []).append(str(f))
    return out


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

    prepared = build_volume(_AsStudy(case["accession"], series), common_spacing)
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

    def __init__(self, accession: str, series: list[dict]) -> None:
        self.accession_number = accession
        self.series = [_AsSeries(s) for s in series]


class _AsSeries:
    def __init__(self, s: dict) -> None:
        self.series_uid = s["uid"]
        self.modality = s["uid"]
        self.image = s["image"]
        self.affine = s["affine"]
        self.source_path = Path(s["path"])
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
