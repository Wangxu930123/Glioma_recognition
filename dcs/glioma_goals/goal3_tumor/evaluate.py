#!/usr/bin/env python
"""目标三 · 病灶识别 的独立评估入口。

用法::

    cd goal3_tumor
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

GOAL = "goal3_tumor"


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
    root = dataset_root(a.data or (cfg.get("data") or {}).get("root"))
    _tr, val = build_datasets(cfg, root, HERE, limit=a.limit,
                              seed=int((cfg.get("train") or {}).get("seed", 42)))

    model = build_model(cfg)
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model_ema") or state.get("model") or state
    model.load_state_dict(sd, strict=False)
    dev = a.device if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.eval().to(dev)

    metrics = evaluate_metrics(model, val, None)
    print(json.dumps({"goal": GOAL, "ckpt": a.ckpt, **metrics}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
