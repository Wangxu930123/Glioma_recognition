"""目标二-B 重复影像检测的数据集（独立副本，可自由修改）。

与其他 Goal 的关键差异：本任务是**配对**训练，而不是单样本分类。

配对从哪来：

1. **官方金标准正对** —— ``labels/2_duplicate.xlsx`` 的 ``(src_img, desc_img)``：
   **两例不同检查的影像实际重复**。这才是比赛要判的东西，也是唯一权威定义。
2. **自监督正对（兜底）**：同一病例做两次不同增强 → 必然相似。
   注意它与官方语义**不是一回事**：那必然相同的图，模型学到的是"增强不变性"，
   而不是"识别重复上传"。没有官方金标准时才用它，且会打印提示。
3. **难负对**：与基准病例模态/序列数接近的**其它**病例。
   随机负对太容易，模型学不到细粒度；难负对才逼它关注真正的差异。

⚠️ **负样本必须避开已知重复对**：若随机抽到的"负样本"恰好是官方标注的重复对，
同一对就被打上两种标签（正 1 / 负 0），模型只会学到噪声。因此负采样显式排除
``dup_of[base]``。

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

    #: 只在**第一份**数据集上提示"缺官方金标准"，避免训练时刷屏
    _warned_no_gold = False

    def __init__(self, cases: list[dict], patch=(96, 96, 96), train: bool = True,
                 common_spacing=(1.0, 1.0, 1.0), cache_dir: Path | None = None,
                 aug_cfg: dict | None = None, seed: int = 42,
                 neg_ratio: int = 1, hard_neg_prob: float = 0.5,
                 global_view_cfg: dict | None = None,
                 gold_pairs: list[tuple[str, str]] | None = None) -> None:
        self.cases = cases
        # ---- 官方重复金标准（``labels/2_duplicate.xlsx``）----
        # 只保留**本 split 里同时存在**的对：划分后某一半可能没有对应病例。
        self.acc2idx = {str(c["accession"]): i for i, c in enumerate(cases)}
        self.gold_pairs = [(a, b) for a, b in (gold_pairs or [])
                           if str(a) in self.acc2idx and str(b) in self.acc2idx]
        # 无向索引，供负样本排除（否则同一对会被打上两种标签）
        self.dup_of: dict[str, set[str]] = {}
        for a, b in self.gold_pairs:
            self.dup_of.setdefault(str(a), set()).add(str(b))
            self.dup_of.setdefault(str(b), set()).add(str(a))
        self.n_gold_used = 0
        if not self.gold_pairs and not DuplicatePairDataset._warned_no_gold and cases:
            DuplicatePairDataset._warned_no_gold = True
            print("[data][告警] 本 split 没有可用的官方重复金标准（2_duplicate.xlsx）→ "
                  "正对退化为「同一病例两次增强」。它与真正的重复影像检测**语义不同**，"
                  "指标会偏乐观；请确认 labels/ 下已放 2_duplicate.xlsx", flush=True)
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

        pa = ga = pb = gb = None
        label = 1.0
        b_acc = str(base["accession"])                            # B 侧来自哪个检查（供核验）

        if is_pos and self.gold_pairs:
            # 正对 ①：**官方金标准** —— 两例不同检查、影像实际重复。
            # 这才是比赛要判的东西；用自增强当正对学到的是"增强不变性"，
            # 两者的判别边界完全不同（指标会明显偏乐观）。
            acc_a, acc_b = self.gold_pairs[k % len(self.gold_pairs)]
            base = self.cases[self.acc2idx[str(acc_a)]]
            second = self.cases[self.acc2idx[str(acc_b)]]
            pa, ga = self._sample_of(base, random.Random(self.rng.getrandbits(32)))
            pb, gb = self._sample_of(second, random.Random(self.rng.getrandbits(32)))
            b_acc = str(acc_b)
            self.n_gold_used += 1
        else:
            # 正对 ②（兜底）：同一病例、同一裁剪区域的两次独立增强
            pa, ga = self._sample_of(base, random.Random(self.rng.getrandbits(32)))
            pb, gb = pa.copy(), (ga.copy() if ga is not None else None)

        if self.train:
            pa, _ = augment(pa, None, random.Random(self.rng.getrandbits(32)), self.aug_cfg)
            pb, _ = augment(pb, None, random.Random(self.rng.getrandbits(32)), self.aug_cfg)
        if gb is not None:
            gb = self._jitter(gb, random.Random(self.rng.getrandbits(32)))

        if not is_pos and len(self.cases) > 1:
            # 负对：**必须避开基准病例已知的重复对**。
            # 否则随机抽到的"负样本"可能正是官方标注的重复影像 ——
            # 同一对被打上两种标签，模型只会学到噪声（且不会有任何报错）。
            dup = self.dup_of.get(str(base["accession"]), set())
            pool = [c for c in self.cases
                    if c["accession"] != base["accession"]
                    and str(c["accession"]) not in dup]
            if pool:
                if self.rng.random() < self.hard_neg_prob:
                    # 难负对：与基准病例序列数接近者（越接近越难分）
                    other = self.rng.choice(
                        sorted(pool, key=lambda c: abs(len(c["series"]) - len(base["series"])))[:8])
                else:
                    other = self.rng.choice(pool)
                pb, gb = self._sample_of(other, random.Random(self.rng.getrandbits(32)))
                if self.train:
                    pb, _ = augment(pb, None, random.Random(self.rng.getrandbits(32)), self.aug_cfg)
                b_acc = str(other["accession"])
                label = 0.0
            # pool 为空（其余病例全是重复对）时保持 label=1.0 且 pb==pa：
            # 宁可少一个负样本，也不制造一个**错误标签**。

        # 键名与 shared.engine 的配对前向约定一致（image / image_b / pair）
        out = {"image": np.ascontiguousarray(pa),
               "image_b": np.ascontiguousarray(pb),
               "pair": np.float32(label),
               "accession": base["accession"],
               # B 侧检查号：让"负样本绝不能是官方重复对"这条不变量**可被断验**
               # （只靠肉眼看代码迟早会漏），也方便排查配对是否按预期取数。
               "accession_b": b_acc}
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

    # ---- 官方重复金标准：``labels/2_duplicate.xlsx``（src_img / desc_img）----
    # 官方把它放在**标注表**里，而不是数据目录下的 csv —— 只扫目录会得到 0 对。
    from shared.official_labels import find_official_labels, read_duplicate_pairs

    gold: list[tuple[str, str]] = []
    dup_file = find_official_labels(data_root).get("duplicate")
    if dup_file:
        gold = read_duplicate_pairs(dup_file)
        print(f"[data] 官方重复金标准 {Path(dup_file).name}：{len(gold)} 对", flush=True)

    train_ds = DuplicatePairDataset(train_cases, patch=patch, train=True,
                                    common_spacing=common, cache_dir=cache,
                                    aug_cfg=cfg, seed=seed, gold_pairs=gold)
    val_ds = DuplicatePairDataset(val_cases, patch=patch, train=False,
                                  common_spacing=common, cache_dir=cache,
                                  aug_cfg=cfg, seed=seed + 1, gold_pairs=gold)
    if gold:
        # 划分后每一半可能只拿到一部分对：说清各自可用多少，
        # 否则"验证集没有正对、指标全是自增强"这件事不会被发现。
        print(f"[data] 重复金标准落入 train={len(train_ds.gold_pairs)} 对 / "
              f"val={len(val_ds.gold_pairs)} 对"
              f"（共 {len(gold)} 对，其余病例不在本 split）", flush=True)
    return train_ds, val_ds
