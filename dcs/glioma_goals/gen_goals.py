#!/usr/bin/env python
"""生成六个 Goal 的独立训练工程（通用部分）。

差异化部分（``dataset.py`` / ``losses.py``）另行单独实现，因为它们承载
各任务真正的算法逻辑；本脚本只负责把**结构相同**的脚手架铺好：
``config.yaml`` / ``train.py`` / ``evaluate.py`` / ``README.md``。

设计要点（直接对应"五个人并行、互不影响"这个需求）：

1. **每个 Goal 的全部产物都落在自己的目录内**（``runs/<tag>/checkpoints``、
   ``runs/<tag>/logs``、``cache``），不存在任何全局共享的写路径；
2. **缓存按 Goal 隔离**：共享缓存看似省磁盘，但两个进程同时写会产生半截文件，
   且损坏后不报错、只让指标莫名变差；
3. **共享库只读**：队员要改预处理/增强/损失，在自己目录里覆盖同名文件，
   不要改 ``shared/``；
4. **训练入口完全独立**：``python train.py`` 即可跑，不需要先跑别人的脚本。
"""
from __future__ import annotations

import pathlib
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent

#: 每个 Goal 的名称、负责人提示、任务说明、默认损失权重
GOALS: dict[str, dict] = {
    "goal1_authenticity": {
        "title": "目标一 · 影像真实性识别",
        "task": "区分真人体 / 假人体 / 非人体，输出检查级 `IsNotHumanBodyProb`",
        "head": "special[0]",
        "weights": {"special": 1.0},
        "extra_deps": [],
    },
    "goal2_stitched": {
        "title": "目标二-A · 拼接影像检测",
        "task": "识别拼接影像，输出检查级 `IsStitchedProb`",
        "head": "special[1]",
        "weights": {"special": 1.0},
        "extra_deps": [],
    },
    "goal2_duplicate": {
        "title": "目标二-B · 重复影像检测",
        "task": "跨检查配对，输出 `duplicate_pairs.jsonl` 的概率",
        "head": "embed",
        "weights": {"embed": 1.0},
        "extra_deps": ["retrieval.py"],
    },
    "goal3_tumor": {
        "title": "目标三 · 病灶识别",
        "task": "区分胶质瘤与非肿瘤性病变，输出 `TumorProbability`（ROC-AUC 评估）",
        "head": "cls[TumorProbability]",
        "weights": {"cls": 1.0},
        "extra_deps": [],
        "warning": "本地数据**无非肿瘤负样本**，该头无法有效训练，AUC 也无法评估；"
                   "需补充脑梗死/脑脓肿病例。",
    },
    "goal4_diagnosis": {
        "title": "目标四 · 辅助诊断",
        "task": "WHO 分级 + 位置/形态/边界 + 坏死囊变出血钙化 + 强化形态 + T2/FLAIR 信号",
        "head": "cls[全部 14 个字段]",
        "weights": {"cls": 1.5},
        "extra_deps": ["labels.py"],
    },
    "goal5_segmentation": {
        "title": "目标五 · 分割与可解释性",
        "task": "T1 增强核心区 + Flair/T2 周围总异常区的二值掩膜",
        "head": "seg[0]=core, seg[1]=peri",
        "weights": {"seg": 1.0, "ds": 0.4},
        "extra_deps": ["postprocess.py"],
    },
}

CONFIG_TMPL = '''# {title} 的独立配置
# ---------------------------------------------------------------------------
# ★ 本文件只属于 {goal}。改这里**不会**影响其他 Goal 的训练。
#   若需要与别人不同的骨干/增强/超参，直接改本文件即可。
# ---------------------------------------------------------------------------
goal: {goal}

# ---- 数据（所有 Goal 共享**只读**的数据根；缓存各自独立）----
data:
  root: /mnt/data_sdb/wangx/data/Brain_MRI/track4_sim     # 可用 --data 覆盖
  common_spacing: [1.0, 1.0, 1.0]                         # 统一到 1mm 公共网格
  cache: cache                                            # 本 Goal 私有缓存目录

# ---- 模型 ----
model:
  arch: mednext
  in_channels: 4
  base: 32
  depth: 4
  blocks_per_stage: 2
  k: 3
  expand: 2
  aniso_z: false
  max_ch: 320

# ---- 训练 ----
train:
  epochs: 100
  batch_size: 2
  lr: 3.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0
  ema_decay: 0.999
  warmup_epochs: 3
  num_workers: 4
  patch: [96, 96, 96]
  seed: 42
  val_ratio: 0.2                                          # 本 Goal 自己划分验证集
  save_every: 10

# ---- 损失（本任务那一路的权重）----
loss_weights: {weights}

# ---- 增强（可按需调整；几何增强会同时作用于影像与掩码）----
augment:
  enabled: true
  flip_prob: 0.5
  affine_prob: 0.3
  max_rot_deg: 12.0
  max_scale: 0.12
  intensity_prob: 0.8
  gamma_range: 0.25
  scale_range: 0.15
  shift_range: 0.15
  noise_prob: 0.3
'''

TRAIN_TMPL = '''#!/usr/bin/env python
"""{title} 的独立训练入口。

用法::

    cd {goal}
    python train.py                       # 使用 config.yaml
    python train.py --tag my_exp          # 自定义实验名（产物落在 runs/my_exp/）
    python train.py --data /path/to/data  # 覆盖数据根
    python train.py --limit 40            # 小规模冒烟（先确认能跑通）
    python train.py --fold 0              # 统一折划分的第 0 折（产物落 runs/<tag>_fold0/）

**并行说明**：本脚本只读写自己目录下的 ``runs/`` 与 ``cache/``，
五个人可以同时在不同 Goal 目录下训练，互不干扰（各自占一张 GPU 即可）：

    CUDA_VISIBLE_DEVICES=0 python train.py     # 你在 goal1 目录
    CUDA_VISIBLE_DEVICES=1 python train.py     # 同事在 goal2 目录
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # 让 `import shared` 可用

import torch                                                        # noqa: E402
from torch.utils.data import DataLoader                             # noqa: E402

from dataset import build_datasets                                  # noqa: E402
from losses import build_loss_fn                                    # noqa: E402
from model import build_model                                       # noqa: E402
from shared.data import available_folds                             # noqa: E402
from shared.engine import TrainConfig, train                        # noqa: E402
from shared.utils import RunPaths, append_jsonl, dataset_root, get_logger, load_yaml  # noqa: E402

GOAL = "{goal}"


def main() -> int:
    ap = argparse.ArgumentParser(description="{title}")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--tag", default=None, help="实验名，默认 <goal>_<时间>")
    ap.add_argument("--data", default=None, help="数据根目录（覆盖 config）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0, help="0=全量；小值用于冒烟测试")
    ap.add_argument("--fold", type=int, default=None,
                    help="统一折划分（folds.json）的折号；不传则用 config 的 train.fold（默认 0）")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    tr = dict(cfg.get("train") or {{}})
    if a.epochs is not None:
        tr["epochs"] = a.epochs
    if a.batch_size is not None:
        tr["batch_size"] = a.batch_size
    if a.lr is not None:
        tr["lr"] = a.lr
    if a.fold is not None:
        tr["fold"] = a.fold

    # ⚠️ 必须**回写** ``cfg["train"]``：折划分发生在 ``build_datasets(cfg, ...)``
    # 内部，它读的是 **cfg**，而 tr 只是它的副本。不回写的话 ``--fold`` 不报错、
    # 但被静默忽略 —— 四个人各训一折，验证集却都在 config 指定的同一折上。
    cfg["train"] = tr

    tag = a.tag or GOAL
    if a.fold is not None:
        # 折号进 tag：否则 ``--fold 0..3`` 会写进**同一个** ``runs/<tag>/``，
        # 后一折覆盖前一折，而训练日志看起来一切正常。
        tag = f"{{tag}}_fold{{a.fold}}"
    paths = RunPaths.for_goal(HERE, tag).ensure()
    log = get_logger(GOAL, paths.log_dir)
    log.info("goal=%s tag=%s device=%s fold=%s", GOAL, tag, a.device, tr.get("fold"))
    log.info("产物目录：%s", paths.root)

    data_root = dataset_root(a.data or (cfg.get("data") or {{}}).get("root"))
    log.info("数据根：%s", data_root)

    # 折号先校验再开跑：写错时 split_train_val 只会**退回按比例划分**，
    # "六个 Goal 同一折"被悄悄破坏，而损失曲线看起来完全正常。
    if a.fold is not None:
        folds_path, keys = available_folds(cfg, data_root, HERE)
        if folds_path is None:
            log.warning("未找到统一折划分（folds.json）→ --fold %s 无效，"
                        "将按 val_ratio 自行划分（各 Goal 的验证集不可比）", a.fold)
        elif keys and str(a.fold) not in keys:
            raise SystemExit(
                f"--fold {{a.fold}} 不在 {{folds_path}} 的折里：可用 {{keys}}。"
                f"折划分由算法工程产出（bash scripts/02_build_dataset.sh）")

    # 数据集与 DataLoader（本 Goal 私有缓存 → 并行安全）
    train_ds, val_ds = build_datasets(cfg, data_root, HERE, limit=a.limit, seed=int(tr.get("seed", 42)))
    log.info("样本数：train=%d val=%d", len(train_ds), len(val_ds))

    def _collate(batch):
        out = {{}}
        for k in batch[0]:
            vals = [b[k] for b in batch]
            if hasattr(vals[0], "shape") or isinstance(vals[0], (int, float)):
                out[k] = torch.as_tensor(torch.stack([torch.as_tensor(v) for v in vals]))
            else:
                out[k] = vals
        return out

    workers = int(tr.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=int(tr.get("batch_size", 2)),
                              shuffle=True, num_workers=workers, collate_fn=_collate,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, int(tr.get("batch_size", 2))),
                            shuffle=False, num_workers=max(1, workers // 2),
                            collate_fn=_collate)

    model = build_model(cfg)
    n_par = sum(p.numel() for p in model.parameters())
    log.info("模型：%s  参数 %.2fM", type(model).__name__, n_par / 1e6)

    tcfg = TrainConfig(epochs=int(tr.get("epochs", 100)),
                       batch_size=int(tr.get("batch_size", 2)),
                       lr=float(tr.get("lr", 3e-4)),
                       weight_decay=float(tr.get("weight_decay", 1e-4)),
                       grad_clip=float(tr.get("grad_clip", 1.0)),
                       ema_decay=float(tr.get("ema_decay", 0.999)),
                       warmup_epochs=int(tr.get("warmup_epochs", 3)),
                       num_workers=workers, patch=tuple(tr.get("patch", [96, 96, 96])),
                       seed=int(tr.get("seed", 42)), out_dir=str(paths.ckpt_dir),
                       tag=tag, save_every=int(tr.get("save_every", 10)))

    res = train(model, train_loader, build_val_fn(train_ds, val_ds),
                tcfg, cfg, build_loss_fn(cfg), device=a.device)
    append_jsonl(paths.root / "history.jsonl", {{"best": res["best"], "ckpt": res["ckpt"]}})
    log.info("完成：best=%s -> %s", res["best"], res["ckpt"])
    return 0


def build_val_fn(train_ds, val_ds):
    """验证函数：默认用**与训练一致的损失**在验证集上算指标。"""
    from losses import evaluate_metrics
    return lambda model: evaluate_metrics(model, val_ds, train_ds)


if __name__ == "__main__":
    raise SystemExit(main())
'''

EVAL_TMPL = '''#!/usr/bin/env python
"""{title} 的独立评估入口。

用法::

    cd {goal}
    python evaluate.py --ckpt runs/<tag>/checkpoints/best.pth

评估口径与比赛对齐：
- **本 Goal 自己划分的验证集**（``--val-ratio``），不做跨 Goal 交叉；
- 指标由本 Goal 的 ``losses.py: evaluate_metrics()`` 定义。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dataset import build_datasets                                  # noqa: E402
from losses import evaluate_metrics                                 # noqa: E402
from model import build_model                                       # noqa: E402
from shared.utils import dataset_root, load_yaml                    # noqa: E402

GOAL = "{goal}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch

    cfg = load_yaml(a.config)
    root = dataset_root(a.data or (cfg.get("data") or {{}}).get("root"))
    _tr, val = build_datasets(cfg, root, HERE, limit=a.limit,
                              seed=int((cfg.get("train") or {{}}).get("seed", 42)))

    model = build_model(cfg)
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model_ema") or state.get("model") or state
    model.load_state_dict(sd, strict=False)
    dev = a.device if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.eval().to(dev)

    metrics = evaluate_metrics(model, val, None)
    print(json.dumps({{"goal": GOAL, "ckpt": a.ckpt, **metrics}}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

README_TMPL = '''# {title}

> 负责人：**待分配**　|　工程目录：`{goal}`

{task}

## 快速开始

```bash
cd {goal}
python train.py --limit 40          # ① 先冒烟：确认数据/依赖就绪（几分钟）
python train.py                     # ② 正式训练（产物在 runs/{goal}/）
python evaluate.py --ckpt runs/{goal}/checkpoints/best.pth
```

## 与其他 Goal **并行训练**（互不影响）

本目录是**完全独立**的训练工程：配置、数据缓存、权重、日志全部落在自己目录内。

```bash
# 五个人同时训练，各占一张卡，互不干扰：
CUDA_VISIBLE_DEVICES=0 python train.py     # 你（本目录）
CUDA_VISIBLE_DEVICES=1 python train.py     # 同事在 goal2_stitched/
CUDA_VISIBLE_DEVICES=2 python train.py     # 同事在 goal3_tumor/
```

唯一共享的是**只读**的 `../shared/` 与数据集本身：

- `../shared/` 是公共库（骨干/数据管线/增强/训练引擎/指标），
  **请不要修改**——改它会影响所有人的实验可比性。需要定制就在本目录里
  覆盖同名文件（`dataset.py` / `losses.py` 本来就是你的独立副本）。
- 数据集只读；缓存写在 `{goal}/cache/`，各 Goal 物理隔离。

## 本目录文件

| 文件 | 作用 | 可改？ |
|---|---|---|
| `config.yaml` | 超参、数据路径、损失权重 | ✅ 随便改 |
| `train.py` | 训练入口 | ✅ |
| `dataset.py` | 本任务的标签构造（继承 `shared.data.BaseCaseDataset`） | ✅ |
| `losses.py` | 本任务的损失与指标 | ✅ |
| `model.py` | 网络（默认引用 `shared` 骨干） | ✅ |
| `evaluate.py` | 独立评估 | ✅ |
| `runs/` `cache/` | 训练产物（自动创建） | — 不要提交 |

## 设计要点

- **1mm 公共网格**：不同序列层厚常不一致（T1C 1mm / FLAIR 3mm），
  不统一网格则多通道无法对齐，掩膜也无法写回原始空间；
- **逐通道 z-score**：MRI 强度无绝对物理意义，跨序列全局归一化会破坏对比；
- **几何增强同时作用于影像与掩码**，且掩码用最近邻重采样
  （线性插值会产生 0.5 这类中间值，污染二值监督信号）；
- **缺模态零占位**并保持通道顺序固定（`t1c, flair, t2, t1`），语义稳定。

## 已知限制

{warning}
'''

GENERIC_WARNING = "无（如有，请在此记录，便于复盘与对外说明）。"


def main() -> int:
    for goal, meta in GOALS.items():
        d = ROOT / goal
        d.mkdir(parents=True, exist_ok=True)
        warn = meta.get("warning", GENERIC_WARNING)
        (d / "config.yaml").write_text(
            CONFIG_TMPL.format(title=meta["title"], goal=goal,
                               weights=str(meta["weights"]).replace("'", ""),
                               ).replace(": false", ": false"), encoding="utf-8")
        (d / "train.py").write_text(
            TRAIN_TMPL.format(title=meta["title"], goal=goal), encoding="utf-8")
        (d / "evaluate.py").write_text(
            EVAL_TMPL.format(title=meta["title"], goal=goal), encoding="utf-8")
        (d / "README.md").write_text(
            README_TMPL.format(title=meta["title"], goal=goal, task=meta["task"],
                               warning=warn), encoding="utf-8")
        print(f"  ✓ {goal:22s} config.yaml / train.py / evaluate.py / README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
