#!/usr/bin/env python
"""目标五 · 分割与可解释性 的独立训练入口。

用法::

    cd goal5_segmentation
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

GOAL = "goal5_segmentation"


def main() -> int:
    ap = argparse.ArgumentParser(description="目标五 · 分割与可解释性")
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
    tr = dict(cfg.get("train") or {})
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
        tag = f"{tag}_fold{a.fold}"
    paths = RunPaths.for_goal(HERE, tag).ensure()
    log = get_logger(GOAL, paths.log_dir)
    log.info("goal=%s tag=%s device=%s fold=%s", GOAL, tag, a.device, tr.get("fold"))
    log.info("产物目录：%s", paths.root)

    data_root = dataset_root(a.data or (cfg.get("data") or {}).get("root"))
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
                f"--fold {a.fold} 不在 {folds_path} 的折里：可用 {keys}。"
                f"折划分由算法工程产出（bash scripts/02_build_dataset.sh）")

    # 数据集与 DataLoader（本 Goal 私有缓存 → 并行安全）
    train_ds, val_ds = build_datasets(cfg, data_root, HERE, limit=a.limit, seed=int(tr.get("seed", 42)))
    log.info("样本数：train=%d val=%d", len(train_ds), len(val_ds))

    def _collate(batch):
        out = {}
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
    append_jsonl(paths.root / "history.jsonl", {"best": res["best"], "ckpt": res["ckpt"]})
    log.info("完成：best=%s -> %s", res["best"], res["ckpt"])
    return 0


def build_val_fn(train_ds, val_ds):
    """验证函数：默认用**与训练一致的损失**在验证集上算指标。"""
    from losses import evaluate_metrics
    return lambda model: evaluate_metrics(model, val_ds, train_ds)


if __name__ == "__main__":
    raise SystemExit(main())
