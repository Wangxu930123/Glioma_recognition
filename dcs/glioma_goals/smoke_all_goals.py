#!/usr/bin/env python
"""六个 Goal 的"能训练"自检：数据集构建 + 单 epoch 训练。

**为什么需要它**：训练链路的失效大多是"跑得完、但没学到"——数据根写错、
掩膜被当成影像读进输入通道、全局头拿到的是 patch 而非整脑视图、权重里
丢掉了 ``cls_spec``……这些都不报错，只会让指标安静地变差。因此这里做的是
**端到端真跑一遍**：建模型 → 取数 → 前向 → 反向 → 保存权重 → 抽查元信息。

两级检查，按需选择：

* ``--datasets-only``：只构建数据集并检查样本键（CPU，几秒钟）；
* 默认：每个 Goal 跑 ``train.py --limit N --epochs 1``，并核对产出的
  ``best.pth`` 里 ``cls_spec`` 与分类头数量一致（需要 GPU）。

用法::

    python smoke_all_goals.py                                  # 用 config 里的数据根
    python smoke_all_goals.py --data /2026aicompetition/datasets/training
    python smoke_all_goals.py --datasets-only --limit 8
    python smoke_all_goals.py --goals goal4_diagnosis,goal1_authenticity

退出码 0 = 六个 Goal 全部通过。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOALS = ("goal1_authenticity", "goal2_stitched", "goal2_duplicate",
         "goal3_tumor", "goal4_diagnosis", "goal5_segmentation")

#: 只有分割头的 Goal 不产整脑视图（它不需要检查级判断）
NO_GLOBAL_VIEW = {"goal5_segmentation"}


def _dataset_probe(goal: str, data: Path | None, limit: int) -> tuple[bool, str]:
    """在**子进程**里构建该 Goal 的数据集并检查样本键。

    用子进程而非同进程导入：六个 Goal 各自都有 ``dataset.py`` / ``model.py``，
    同进程连续导入会互相污染 ``sys.modules``，得到的结论不可信。
    """
    data_literal = "None" if data is None else repr(str(data))
    code = f'''
import importlib
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, {str(HERE)!r})
gdir = Path({str(HERE / goal)!r})
sys.path.insert(0, str(gdir))

cfg = yaml.safe_load((gdir / "config.yaml").read_text(encoding="utf-8")) or {{}}
root = Path({data_literal}) if {data_literal} is not None else \\
    Path((cfg.get("data") or {{}}).get("root") or ".")
ds = importlib.import_module("dataset")
tr, va = ds.build_datasets(cfg, root, gdir, limit={limit}, seed=42)
assert len(tr) > 0 and len(va) > 0, f"train={{len(tr)}} val={{len(va)}}（数据根 {{root}}）"
item = tr[0]
img = np.asarray(item["image"])
assert img.ndim == 4, f"image{{img.shape}}"
gv = item.get("image_global")
if {goal!r} in {sorted(NO_GLOBAL_VIEW)!r}:
    assert gv is None, "该 Goal 不应产出整脑视图"
else:
    assert gv is not None and np.asarray(gv).ndim == 4, f"缺 image_global: {{gv}}"
if "image_b" in item:
    assert "image_global_b" in item and "pair" in item, "配对分支键缺失"
print(f"root={{root}} train={{len(tr)}} val={{len(va)}} image={{tuple(img.shape)}} "
      f"global={{None if gv is None else tuple(np.asarray(gv).shape)}}")
'''
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(HERE / goal))
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        return False, lines[-1] if lines else "未知错误"
    return True, proc.stdout.strip().splitlines()[-1]


def _weight_probe(goal: str, tag: str) -> tuple[bool, str]:
    """核对训练产出的权重元信息：``cls_spec`` 必须与分类头数量一致。

    这是推理侧把"头"映射到"字段"的唯一依据；缺了它推理只能按位置猜，
    顺序一错，所有结构化字段静默错位。
    """
    code = f'''
import torch
ck = torch.load({str(HERE / goal / "runs" / tag / "checkpoints" / "best.pth")!r},
                map_location="cpu", weights_only=False)
spec = ck.get("cls_spec") or []
state = ck.get("model") or ck
n_heads = sum(1 for k in state if k.startswith("cls_heads.") and k.endswith(".weight"))
assert spec, "cls_spec 为空：推理侧无法把分类头对应到字段"
assert len(spec) == n_heads, f"cls_spec={{len(spec)}} 但权重里有 {{n_heads}} 个分类头"
assert ck.get("model_cfg"), "model_cfg 为空：推理会按默认结构重建，形状不符的层会被静默跳过"
names = [n for n, _ in spec]
print("cls_spec=" + ",".join(names[:3]) + ("…" if len(names) > 3 else "")
      + f" 共 {{len(spec)}} 个")
'''
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(HERE / goal))
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        return False, lines[-1] if lines else "未知错误"
    return True, proc.stdout.strip().splitlines()[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="六个 Goal 的能训练自检")
    ap.add_argument("--data", default=os.environ.get("GLIOMA_GOALS_DATA"),
                    help="数据根（默认取各 Goal config.yaml 的 data.root）")
    ap.add_argument("--limit", type=int, default=6, help="抽样病例数（0=全量）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--tag", default="smoke_pl", help="实验名（不与正式产物混用）")
    ap.add_argument("--goals", default="", help="逗号分隔，只跑指定 Goal")
    ap.add_argument("--datasets-only", action="store_true", help="只构建数据集（CPU，快）")
    a = ap.parse_args()

    goals = tuple(g.strip() for g in a.goals.split(",") if g.strip()) or GOALS
    data = Path(a.data) if a.data else None
    results: list[tuple[str, bool, str]] = []

    for goal in goals:
        print(f"\n=== {goal} ===", flush=True)
        # 不给 --data 时由各 Goal 的 config.yaml 决定数据根（与 train.py 一致）
        ok, detail = _dataset_probe(goal, data, a.limit)
        print(f"  [{'PASS' if ok else 'FAIL'}] 数据集 {detail}", flush=True)
        if ok and not a.datasets_only:
            cmd = [sys.executable, "train.py", "--limit", str(a.limit),
                   "--epochs", str(a.epochs), "--tag", a.tag, "--device", "cuda"]
            if data:
                cmd += ["--data", str(data)]
            proc = subprocess.run(cmd, cwd=str(HERE / goal),
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
                ok, detail = False, f"训练失败：{tail[0] if tail else '未知错误'}"
            else:
                ok, detail = _weight_probe(goal, a.tag)
            print(f"  [{'PASS' if ok else 'FAIL'}] 训练+权重元信息 {detail}", flush=True)
        results.append((goal, ok, detail))

    print("\n" + "=" * 60)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for goal, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {goal:<20} {detail if not ok else ''}")
    print(f"通过 {n_ok}/{len(results)}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
