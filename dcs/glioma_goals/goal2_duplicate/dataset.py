"""目标二-B 重复影像检测的数据集（独立副本，可自由修改）。

与其他 Goal 的关键差异：本任务是**配对**训练，而不是单样本分类。

配对从哪来（金标准只有十几对，远不足以训练）：

1. **自监督正对**：同一病例做**两次不同增强** → 必然相似 → 正对。
   这是重复影像任务的主力监督信号，因为"重复上传/前后复查"的本质就是
   "同一检查的两种呈现"；
2. **难负对**：**同一患者不同检查**（或增强参数接近的不同病例）→ 负对。
   随机负对太容易，模型学不到细粒度；难负对才逼它关注真正的差异；
3. **金标准对**（若 ``label.json`` 提供了重复关系）→ 最可靠，用于微调。

⚠️ 训练与推理的**尺度必须一致**：训练用 1mm 公共网格的整脑视图，
   推理（``task.py`` 的 ``update()``）也必须用同一尺度与同一视图裁剪参数。
   尺度不一致会让嵌入分布漂移，表现为"训练时相似度分得很开、上线后全乱"。
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from shared.data import (BaseCaseDataset, augment, crop_patch, discover_cases,
                         load_case, lesion_center, split_train_val)

GOAL = "goal2_duplicate"


def split_cases(cases: list[dict], val_ratio: float, seed: int):
    """按 accession 稳定划分（与其余 Goal 相同的口径）。"""
    idx = list(range(len(cases)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(len(cases) * float(val_ratio))))
    val_idx = set(idx[:n_val])
    return ([c for i, c in enumerate(cases) if i not in val_idx],
            [c for i, c in enumerate(cases) if i in val_idx])


class DuplicatePairDataset:
    """配对数据集：每个样本返回 ``(image_a, image_b, pair_label)``。

    ``image_a`` 与 ``image_b`` 是**同一个裁剪区域**的两种增强，
    因此正对在语义上确实"应该相似"，不会引入错误监督。
    """

    def __init__(self, cases: list[dict], patch=(96, 96, 96), train: bool = True,
                 common_spacing=(1.0, 1.0, 1.0), cache_dir: Path | None = None,
                 aug_cfg: dict | None = None, seed: int = 42,
                 neg_ratio: int = 1, hard_neg_prob: float = 0.5,
                 global_view_cfg: dict | None = None) -> None:
        self.cases = cases
        self.patch = tuple(patch)
        self.train = train
        self.common_spacing = common_spacing
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.aug_cfg = aug_cfg or {}
        self.rng = random.Random(seed)
        self.neg_ratio = max(1, int(neg_ratio))
        self.hard_neg_prob = float(hard_neg_prob)
        #: 整脑视图配置。**嵌入头必须用整脑视图**——本文件开头的约定
        #: （"训练用 1mm 公共网格的整脑视图，推理也必须用同一尺度"）
        #: 早期只写进了注释、没有落到代码：实现走的是 ``crop_patch``，
        #: 于是训练看到 96mm patch、推理看到 192mm 整脑，嵌入分布整体漂移。
        #: 取值优先级：显式参数 > ``config.yaml`` 的 ``global_view`` 段。
        self.gv = dict(global_view_cfg if global_view_cfg is not None
                       else (aug_cfg or {}).get("global_view") or {})

    def __len__(self) -> int:
        return max(1, len(self.cases) * (1 + self.neg_ratio))

    # ------------------------------------------------------------------ #
    def _sample_of(self, case: dict, rng: random.Random
                   ) -> tuple[np.ndarray, np.ndarray | None]:
        """取该病例的 ``(patch, 整脑视图)``。

        两者共用**同一次** ``load_case`` 结果，避免重复读盘/重采样
        （配对任务每个样本要取 1~2 个病例，重复读盘的代价会翻倍）。
        """
        vol, masks, _ = load_case(case, self.common_spacing, self.cache_dir)
        center = lesion_center(masks) if self.train else None
        pv, _ = crop_patch(vol, self.patch, rng, center, 0.7 if self.train else 0.0)

        gv = None
        if self.gv.get("enabled", True):
            from shared.volume import global_view
            gv = global_view(vol, size_mm=float(self.gv.get("size_mm", 192)),
                             out=int(self.gv.get("out", 96)),
                             spacing=float(self.common_spacing[0]), center=center)
        return pv, gv

    def _jitter(self, g: np.ndarray, rng: random.Random) -> np.ndarray:
        """整脑视图的**强度**抖动（刻意不做几何变换）。

        正对需要"同一区域、两种呈现"，但几何变换会改变物理尺度，
        破坏"训练 == 推理"这一前提；重复影像的实际差异恰恰主要来自
        重建/噪声而非几何，因此只扰动强度。
        """
        if not self.train:
            return g
        scale = 1.0 + rng.uniform(-0.08, 0.08)
        shift = rng.uniform(-0.08, 0.08)
        return (g * scale + shift).astype(np.float32)

    def __getitem__(self, i: int) -> dict:
        if not self.cases:
            z = np.zeros((4,) + self.patch, np.float32)
            out = {"image": z, "image_b": z.copy(), "pair": np.float32(1.0)}
            if self.gv.get("enabled", True):
                n = int(self.gv.get("out", 96))
                g = np.zeros((4, n, n, n), np.float32)
                out["image_global"], out["image_global_b"] = g, g.copy()
            return out

        k = i // (1 + self.neg_ratio)
        base = self.cases[k % len(self.cases)]
        is_pos = (i % (1 + self.neg_ratio)) == 0

        # 正对：同一病例、**同一裁剪区域**的两次独立增强
        rng_patch = random.Random(self.rng.getrandbits(32))
        pa, ga = self._sample_of(base, rng_patch)
        pb, gb = pa.copy(), (ga.copy() if ga is not None else None)
        if self.train:
            pa, _ = augment(pa, None, random.Random(self.rng.getrandbits(32)), self.aug_cfg)
            pb, _ = augment(pb, None, random.Random(self.rng.getrandbits(32)), self.aug_cfg)
        if gb is not None:
            gb = self._jitter(gb, random.Random(self.rng.getrandbits(32)))

        label = 1.0
        if not is_pos and len(self.cases) > 1:
            # 负对：优先取"难负对"（与基准病例模态/层厚接近者）
            if self.rng.random() < self.hard_neg_prob:
                cand = sorted(self.cases, key=lambda c: abs(len(c["series"]) - len(base["series"])))
                cand = [c for c in cand if c["accession"] != base["accession"]][:8]
                other = self.rng.choice(cand) if cand else self.rng.choice(
                    [c for c in self.cases if c["accession"] != base["accession"]])
            else:
                other = self.rng.choice([c for c in self.cases
                                         if c["accession"] != base["accession"]])
            pb, gb = self._sample_of(other, random.Random(self.rng.getrandbits(32)))
            if self.train:
                pb, _ = augment(pb, None, random.Random(self.rng.getrandbits(32)), self.aug_cfg)
            label = 0.0

        # 键名与 shared.engine 的配对前向约定一致（image / image_b / pair）
        out = {"image": np.ascontiguousarray(pa),
               "image_b": np.ascontiguousarray(pb),
               "pair": np.float32(label),
               "accession": base["accession"]}
        if ga is not None:
            out["image_global"] = np.ascontiguousarray(ga)
            out["image_global_b"] = np.ascontiguousarray(gb if gb is not None else ga)
        return out


def build_datasets(cfg: dict, data_root: Path, goal_dir: Path, limit: int = 0,
                   seed: int = 42):
    """构建 ``(train_ds, val_ds)``；缓存写在**本 Goal 目录**下。"""
    tr = cfg.get("train") or {}
    dc = cfg.get("data") or {}
    cases = discover_cases(data_root, limit=limit or None)
    if len(cases) < 2:
        raise RuntimeError(f"重复影像任务至少需要 2 个检查，实际 {len(cases)} 个")
    # 验证集优先用**统一折划分**（folds.json）。
    # 这条与"五个 Goal 是由五个人并行做、还是一个人串行做"**无关**：
    #   · 一个人串行做时，统一口径才让五个指标互相可比；
    #   · 五个人并行做时，统一口径才让各 Goal 的权重可以合并/集成。
    # 仅当折划分不存在、或与当前数据对不上时才回退到 val_ratio（并有日志提示）。
    train_cases, val_cases, _split_src = split_train_val(
        cases, cfg, data_root, goal_dir, seed)

    cache = (goal_dir / str(dc.get("cache", "cache"))).resolve()
    common = tuple(dc.get("common_spacing", [1.0, 1.0, 1.0]))
    patch = tuple(tr.get("patch", [96, 96, 96]))
    return (
        DuplicatePairDataset(train_cases, patch=patch, train=True, common_spacing=common,
                             cache_dir=cache, aug_cfg=cfg, seed=seed),
        DuplicatePairDataset(val_cases, patch=patch, train=False, common_spacing=common,
                             cache_dir=cache, aug_cfg=cfg, seed=seed + 1),
    )
