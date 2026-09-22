#!/usr/bin/env python
"""目标三 · 病灶识别 的独立训练入口。

用法::

    cd goal3_tumor
    python train.py                       # 使用 config.yaml
    python train.py --tag my_exp          # 自定义实验名（产物落在 runs/my_exp/）
    python train.py --data /path/to/data  # 覆盖数据根
    python train.py --limit 40            # 小规模冒烟（先确认能跑通）

**并行说明**：本脚本只读写自己目录下的 ``runs/`` 与 ``cache/``，
五个人可以同时在不同 Goal 目录下训练，互不干扰（各自占一张 GPU 即可）：

    CUDA_VISIBLE_DEVICES=0 python train.py     # 你在 goal1 目录
    CUDA_VISIBLE_DEVICES=1 python train.py     # 同事在 goal2 目录
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # 让 `import shared` 可用

import torch                                                        # noqa: E402
from torch.utils.data import DataLoader                             # noqa: E402

from dataset import build_datasets                                  # noqa: E402
from losses import build_loss_fn                                    # noqa: E402
from model import build_model                                       # noqa: E402
from shared.engine import TrainConfig, train                        # noqa: E402
from shared.utils import RunPaths, append_jsonl, dataset_root, get_logger, load_yaml  # noqa: E402

GOAL = "goal3_tumor"


def main() -> int:
    ap = argparse.ArgumentParser(description="目标三 · 病灶识别")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--tag", default=None, help="实验名，默认 <goal>_<时间>")
    ap.add_argument("--data", default=None, help="数据根目录（覆盖 config）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0, help="0=全量；小值用于冒烟测试")
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

    tag = a.tag or GOAL
    paths = RunPaths.for_goal(HERE, tag).ensure()
    log = get_logger(GOAL, paths.log_dir)
    log.info("goal=%s tag=%s device=%s", GOAL, tag, a.device)
    log.info("产物目录：%s", paths.root)

    data_root = dataset_root(a.data or (cfg.get("data") or {}).get("root"))
    log.info("数据根：%s", data_root)

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
