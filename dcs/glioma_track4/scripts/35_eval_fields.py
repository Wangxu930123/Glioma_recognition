#!/usr/bin/env python
"""字段级分类评估（目标③④）：在指定 val 上跑**全局头**，输出 14 个字段的
accuracy / precision / recall / F1 / AUC（二分类）与 accuracy / macro-F1（多分类）。

为什么需要它（工程原有的三个评估口都不覆盖字段）：
- `trainer.validate()` 只算**分割** Dice → 训练日志里永远看不到分类指标；
- `04_eval.sh` / `src/evaluation/evaluate.py` → 分割（Dice/NSD/HD95）+ 重复影像；
- `15_eval_special_dup.py` → 目标①②（假人体/拼接）+ 重复影像；
- `13/14/16` 是缓存 / 阈值标定 / 收尾，都不算字段指标。
所以本脚本是**唯一的**目标③④本地评估口径。

口径与训练**完全同源**（这点最关键，否则数字没意义）：
- 病例与金标准：走 `GliomaDataset(train=False)`，即与训练同一个 `labels.yaml` 字段顺序、
  同一套「case → 标签索引」转换（含 `Location` 多标签取首个匹配项的处理）；
- 前向：与推理同一尺度（`global_view` 整脑视图，`pre.size_mm/out`），同一 TTA 组合；
- 激活：`_forward_batch` 内部已做 sigmoid(二分类) / softmax(多分类)，这里直接用概率。

⚠️ 目标③（`TumorProbability` 的 ROC-AUC）**需要负样本**（非肿瘤性病变）。
   官方训练/验证集里若没有 `abn` 病例，本脚本会打印"AUC 不适用（无负样本）"——
   这是数据决定的（见 `Glioma_recognition-main/tasks/goal3_tumor/config.py` 的已知限制），
   不是脚本缺陷。

用法：
    # 先小样本自检（20 例，几十秒），确认链路通了再看全量
    python3 scripts/35_eval_fields.py --split fold --fold 0 --limit 20

    # 全量（折内 val，666 例）
    python3 scripts/35_eval_fields.py --split fold --fold 0

    # 显式指定权重（默认自动找 checkpoints/g4_fold<N>/best.pth）
    python3 scripts/35_eval_fields.py --split fold --fold 0 --ckpt checkpoints/g4_fold0/best.pth

产物：stdout 表格 + `logs/eval_fields_<split><fold>.json`（可直接贴进报告）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------- #
# 指标（只依赖 numpy；与 src/evaluation/metrics.py 的分割指标无关）
# --------------------------------------------------------------------------- #
def auc(scores, labels) -> float:
    """秩和法 ROC-AUC（并列取平均秩）。任一类别缺失时返回 nan。"""
    s = np.asarray(scores, float)
    y = np.asarray(labels, float) > 0.5
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sr = s[order]
    ranks = np.empty(len(sr), float)
    i = 0
    while i < len(sr):                                     # 并列值 → 平均秩
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[y[order]].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(probs, labels, thr: float = 0.5) -> dict:
    """二分类：accuracy / precision / recall / F1 / AUC / 正样本数。"""
    p = np.asarray(probs, float)
    y = np.asarray(labels, float) > 0.5
    pred = p >= thr
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "n": int(len(p)), "n_pos": int(y.sum()), "thr": thr,
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(1e-9, prec + rec),
        "auc": auc(p, y),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def macro_f1(true, pred, n_cls: int) -> float:
    """多分类 macro-F1（只统计在真值里出现过的类别，避免从未出现的类把均值拉低）。"""
    f1s = []
    for c in range(n_cls):
        tp = sum(1 for t, q in zip(true, pred) if t == c and q == c)
        fp = sum(1 for t, q in zip(true, pred) if t != c and q == c)
        fn = sum(1 for t, q in zip(true, pred) if t == c and q != c)
        if tp + fp + fn == 0:
            continue
        pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f1s.append(2 * pr * rc / max(1e-9, pr + rc))
    return float(np.mean(f1s)) if f1s else float("nan")


def multiclass_metrics(preds, trues, n_cls: int) -> dict:
    p = [int(x) for x in preds]
    t = [int(x) for x in trues]
    return {"n": len(p), "accuracy": float(np.mean([a == b for a, b in zip(p, t)])),
            "macro_f1": macro_f1(t, p, n_cls),
            "n_class_seen": len(set(t))}


# --------------------------------------------------------------------------- #
def _fmt(v, nd: int = 4) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "  nan" if not np.isfinite(f) else f"{f:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=("auto", "fold", "external"), default="fold",
                    help="fold=折内 val（**有字段金标准**，本脚本的默认口径）；"
                         "external=官方验证集（**没有字段金标准表**，会全部跳过）")
    ap.add_argument("--ckpt", default=None, help="逗号分隔即为多折集成")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部；先跑 20 例自检")
    ap.add_argument("--tta", default=None, help="TTA 轴，逗号分隔（默认取 preprocess.yaml）")
    a = ap.parse_args()

    from src.data.dataset import GliomaDataset
    from src.inference.pipeline import GliomaPipeline
    from src.inference.sliding import _forward_batch, _tta_combos
    from src.utils.config import (external_val_manifest, load_config, load_paths, resolve)

    paths = load_paths()
    split = a.split
    if split == "auto":
        split = "external" if external_val_manifest()[0] is not None else "fold"

    # ---------------- 取病例（与训练同一份 val）----------------
    if split == "external":
        man, mp = external_val_manifest()
        if man is None:
            raise SystemExit(f"[fields] 验证集清单不可用：{mp}\n"
                             f"  · 生成：export VAL_ROOT=... && bash scripts/01_probe.sh --val")
        cases = [c for c in man["cases"] if c.get("images")]
        print(f"[fields] ⚠️ 官方验证集没有字段金标准表 → 绝大多数病例会被跳过；"
              f"要看字段指标请用 --split fold（清单 {mp}）", flush=True)
    else:
        with open(resolve(paths["folds"]), encoding="utf-8") as f:
            folds = json.load(f)
        with open(resolve(paths["manifest"]), encoding="utf-8") as f:
            man = json.load(f)
        by = {c["accession"]: c for c in man["cases"]}
        cases = [by[x] for x in folds[str(a.fold)]["val"] if x in by]
        print(f"[fields] 口径 = fold{a.fold} 折内 val（{len(cases)} 例，来自训练集清单）",
              flush=True)
    if a.limit:
        cases = cases[: a.limit]
    if not cases:
        print("[fields] ✗ 没有病例", flush=True)
        return 1

    # ---------------- 权重 ----------------
    raw = a.ckpt or resolve(os.path.join(paths["checkpoints_dir"],
                                         f"g4_fold{a.fold}", "best.pth"))
    ckpts = [c.strip() if os.path.isabs(c.strip()) else resolve(c.strip())
             for c in str(raw).split(",") if c.strip()]
    for c in ckpts:
        if not os.path.exists(c):
            raise SystemExit(f"[fields] 权重不存在：{c}")
    print(f"[fields] 权重 {len(ckpts)} 个：" + "，".join(ckpts), flush=True)

    pre = load_config("preprocess.yaml")
    tcfg = load_config("train.yaml")
    fields = load_config("labels.yaml")["fields"]

    pipe = GliomaPipeline(ckpts)
    dev = pipe.device
    tta = tuple(x.strip() for x in a.tta.split(",")) if a.tta else \
        tuple(pre["inference"].get("tta_flips") or ())
    combos = _tta_combos(tta)
    tta_batch = int(pre["inference"].get("tta_batch", 2))
    amp = torch.bfloat16
    print(f"[fields] TTA 轴={tta} → {len(combos)} 组合 | tta_batch={tta_batch} | device={dev}",
          flush=True)

    # ---------------- 金标准 + 整脑视图：走训练同一个 Dataset ----------------
    # 这样"case → 标签索引"的转换与 `GliomaDataset.__getitem__` **同源**，
    # 不会出现"评估口径与训练口径不一致"的隐性错误。
    ds = GliomaDataset(cases, train=False, patch=tuple(pre["inference"]["patch"]),
                       pre_cfg=pre, label_fields=fields, aug_cfg=tcfg)

    bin_rec: dict[str, dict] = defaultdict(lambda: {"p": [], "y": []})
    one_rec: dict[str, dict] = defaultdict(lambda: {"pred": [], "true": []})
    skipped: dict[str, int] = defaultdict(int)
    done = 0
    for i in range(len(ds)):
        b = ds[i]
        y = b["labels"].numpy().astype(float)
        m = b["label_mask"].numpy().astype(bool)
        if not m.any():                                    # 该例一个字段金标准都没有
            for f in fields:
                skipped[f["key"]] += 1
            continue
        x = b["whole"][None].to(dev, torch.float32)
        try:
            r = _forward_batch(pipe.models, x, combos, amp, tta_batch=tta_batch)
        except Exception as e:                             # noqa: BLE001
            print(f"[fields] ✗ {b.get('accession')}: {e}", flush=True)
            continue
        cls = r["cls"]
        if len(cls) < len(fields):
            print(f"[fields] ⚠️ 权重只有 {len(cls)} 个分类头，少于 labels.yaml 的 "
                  f"{len(fields)} 个字段 → 后续字段将被跳过", flush=True)
        if i == 0:                                         # 形状自检（只打一次，便于排错）
            print("[fields] 头输出形状自检: " + ", ".join(
                f"{fields[k]['key']}({fields[k]['type']})={tuple(cls[k].shape)}"
                for k in range(min(3, len(cls)))), flush=True)
        for fi, f in enumerate(fields):
            if not m[fi] or fi >= len(cls):
                if not m[fi]:
                    skipped[f["key"]] += 1
                continue
            # ⚠️ `_forward_batch` 返回的 cls[fi] 是**已按样本/TTA/模型平均过**的向量：
            #    二分类 → 形状 (1,)，多分类 → (n_cls,)
            #    （`inference/sliding.py` 里 `cls = [torch.cat(v, dim=0).mean(0) ...]`）。
            #    所以**不能**再写 `cls[fi][0]` —— 那会取成 0 维标量，`p[0]` 直接 IndexError。
            #    这里统一 ravel 成 1 维；B>1 时也只取第 0 个样本（本脚本恒为单样本前向）。
            t = cls[fi]
            if t.dim() >= 2:
                t = t[0]
            p = t.float().cpu().numpy().reshape(-1)
            if f["type"] == "binary":
                bin_rec[f["key"]]["p"].append(float(p[0]))
                bin_rec[f["key"]]["y"].append(float(y[fi]))
            else:
                one_rec[f["key"]]["pred"].append(int(np.argmax(p)))
                one_rec[f["key"]]["true"].append(int(y[fi]))
        done += 1
        if (i + 1) % 50 == 0:
            print(f"[fields] {i + 1}/{len(ds)} …", flush=True)

    if done == 0:
        print("[fields] ✗ 没有一例带字段金标准 —— 折内 val 才有；"
              "官方验证集不提供字段金标准（见脚本头注释）", flush=True)
        return 1

    # ---------------- 汇总 ----------------
    rep: dict = {"split": split, "fold": a.fold if split != "external" else None,
                 "ckpt": ckpts, "n_cases": done, "tta": list(tta), "fields": {}}
    print(f"\n[fields] 完成 {done} 例 | 字段指标（二分类阈值 0.5）", flush=True)
    print("-" * 96, flush=True)
    print(f"{'字段':<20}{'类型':<9}{'n':>5}{'正例':>6}  "
          f"{'accuracy':>9}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}", flush=True)
    print("-" * 96, flush=True)

    bin_f1, one_acc = [], []
    for f in fields:
        k, t = f["key"], f["type"]
        if t == "binary":
            rec = bin_rec.get(k)
            if not rec or not rec["p"]:
                print(f"{k:<20}{t:<9}{0:>5}{'-':>6}  （无金标准，跳过）", flush=True)
                continue
            mm = binary_metrics(rec["p"], rec["y"])
            n_cls = 1
            if not np.isfinite(mm["auc"]):
                note = " ← 无负样本，AUC 不适用" if mm["n_pos"] == mm["n"] else ""
            else:
                note = ""
            print(f"{k:<20}{t:<9}{mm['n']:>5}{mm['n_pos']:>6}  "
                  f"{_fmt(mm['accuracy']):>9}{_fmt(mm['precision']):>10}"
                  f"{_fmt(mm['recall']):>8}{_fmt(mm['f1']):>8}{_fmt(mm['auc']):>8}{note}",
                  flush=True)
            rep["fields"][k] = {**mm, "type": t}
            if np.isfinite(mm["f1"]):
                bin_f1.append(mm["f1"])
        else:
            rec = one_rec.get(k)
            n_cls = len(f.get("classes") or [])
            if not rec or not rec["pred"]:
                print(f"{k:<20}{t:<9}{0:>5}{'-':>6}  （无金标准，跳过）", flush=True)
                continue
            mm = multiclass_metrics(rec["pred"], rec["true"], n_cls)
            print(f"{k:<20}{t:<9}{mm['n']:>5}{mm['n_class_seen']:>3}类  "
                  f"{_fmt(mm['accuracy']):>9}{'—':>10}{'—':>8}{_fmt(mm['macro_f1']):>8}"
                  f"{'—':>8}   macro-F1", flush=True)
            rep["fields"][k] = {**mm, "type": t, "n_classes": n_cls}
            if np.isfinite(mm["accuracy"]):
                one_acc.append(mm["accuracy"])

    print("-" * 96, flush=True)
    rep["macro_binary_f1"] = float(np.mean(bin_f1)) if bin_f1 else float("nan")
    rep["macro_single_accuracy"] = float(np.mean(one_acc)) if one_acc else float("nan")
    print(f"[fields] 汇总：二分类字段 macro-F1 = {_fmt(rep['macro_binary_f1'])}"
          f"（{len(bin_f1)} 个字段） | 多分类字段 macro-accuracy = "
          f"{_fmt(rep['macro_single_accuracy'])}（{len(one_acc)} 个字段）", flush=True)
    rep["n_skipped_by_field"] = dict(skipped)
    tp = rep["fields"].get("TumorProbability")
    if tp is not None and not np.isfinite(tp.get("auc", float("nan"))):
        print("[fields] ⚠️ 目标③ TumorProbability：本 val 内全是正样本（没有非肿瘤性病变）"
              "→ ROC-AUC 无法计算。这是数据决定的（见 tasks/goal3_tumor/config.py），"
              "补齐 abn 负样本后即可启用。", flush=True)

    tag = f"{split}{a.fold}" if split != "external" else "external"
    out = resolve(os.path.join(paths["logs_dir"], f"eval_fields_{tag}.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print(f"\n[fields] 已写出：{out}", flush=True)
    print("[fields] 提示：这只是**单权重评折内 val**（666 例的 1/5 量级），"
          "不是最终口径；最终数字看验证集/多折集成。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
