#!/usr/bin/env python
"""目标一/二专项评估：假人体、拼接影像、重复影像。

只跑**全局头**（整脑视图一次前向），不需要滑窗，因此很快（约 0.5s/例）。

## 为什么默认是 OOF 评估

本任务的正样本极稀疏（实测 624 例中 fake 6、Composition 6、重复金标准 12 对）。
若按"某一折的验证集"来评，单折 val 内只有 1~2 个正样本、重复对甚至为 0 —— 
算出的 AUC 会在 0/0.5/1.0 之间跳变，**没有统计意义**；而若直接跑全量，
折模型又见过自己训练集里的病例 → **AUC 虚高**（数据泄漏）。

因此默认采用 **OOF（out-of-fold）**：逐折用该折的模型只预测**本折验证集**，
再把各折结果汇总。每一例都恰好被"没见过它的那个折模型"预测过一次，
既**无泄漏**，又用满全部样本，让稀疏正样本尽可能参与统计。

用法::

    python scripts/15_eval_special_dup.py                    # OOF（推荐，无泄漏）
    python scripts/15_eval_special_dup.py --all              # 全量单次（含训练集，仅看趋势）
    python scripts/15_eval_special_dup.py --ckpt a.pt,b.pt   # 指定权重（--all 时=集成）
    python scripts/15_eval_special_dup.py --limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC（Mann-Whitney U 统计量，含并列处理）。"""
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r = ranks[: len(pos)].sum()
    return float((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _split_ckpts(raw: str, paths) -> list[str]:
    from src.utils.config import resolve
    out: list[str] = []
    for c in str(raw).split(","):
        c = c.strip()
        if c:
            out.append(c if os.path.isabs(c) else resolve(c))
    for c in out:
        if not os.path.exists(c):
            raise SystemExit(f"[eval-sd] 权重不存在: {c}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="构造默认权重路径时使用的折号")
    ap.add_argument("--ckpt", default=None, help="权重路径；逗号分隔即多折集成（--all 时生效）")
    ap.add_argument("--limit", type=int, default=0, help="0=全量")
    ap.add_argument("--fp-weight", type=float, default=0.5)
    ap.add_argument("--oof", dest="oof", action="store_true", default=True,
                    help="（默认）OOF 评估：逐折用本折模型评本折 val，汇总后无泄漏")
    ap.add_argument("--all", dest="oof", action="store_false",
                    help="全量单次评估（含训练集 → 数字虚高，仅用于快速看趋势）")
    a = ap.parse_args()

    from src.data.dataset import brain_center, global_view, load_case_cached, resolve_cache_dir
    from src.inference.duplicate import (calibrate, fingerprint_similarity, gold_similarities,
                                         match_pairs)
    from src.inference.pipeline import GliomaPipeline
    from src.inference.sliding import _forward_batch, _tta_combos
    from src.inference.writer import case_fingerprint
    from src.evaluation import metrics as M
    from src.utils.config import load_config, load_paths, resolve

    paths = load_paths()
    ckpts = _split_ckpts(
        a.ckpt or os.path.join(paths["checkpoints_dir"], f"g4_fold{a.fold}", "best.pth"), paths)

    with open(resolve(paths["manifest"]), encoding="utf-8") as f:
        man = json.load(f)
    cases_all = [c for c in man["cases"] if c.get("images")]
    by_acc = {c["accession"]: c for c in cases_all}
    folds: dict = {}
    _fp = resolve(paths["folds"])
    if os.path.exists(_fp):
        with open(_fp, encoding="utf-8") as f:
            folds = json.load(f)

    special = man.get("special") or {}
    pos_fake = set(special.get("fake_cases") or [])
    pos_comp = set(special.get("composition_cases") or [])
    gold = {tuple(sorted(p)) for p in (special.get("gold_pairs") or [])}

    # ---------------- 组装评估范围 ----------------
    # 每个 scope = (标签, 权重列表, 该 scope 内的 AccessionNumber 集合)
    scopes: list[tuple[str, list[str], list[str]]] = []
    if a.oof and folds:
        # 去重：同一病例若出现在多个折的 val（旧版 build_folds 的轮转绕回缺陷），
        # 只保留首次出现，避免同一例被多个折模型重复评估、污染统计。
        _seen: set[str] = set()
        _dup_cnt = 0
        for f in sorted(folds, key=lambda x: int(x)):
            p = resolve(os.path.join(paths["checkpoints_dir"], f"g4_fold{f}", "best.pth"))
            if not os.path.exists(p):
                print(f"[eval-sd] 跳过 fold{f}（缺 {p}）")
                continue
            accs = []
            for x in (folds[f].get("val") or []):
                if x not in by_acc:
                    continue
                if x in _seen:
                    _dup_cnt += 1
                    continue
                _seen.add(x)
                accs.append(x)
            if accs:
                scopes.append((f"fold{f}", [p], accs))
        if _dup_cnt:
            print(f"[eval-sd] ⚠️ {_dup_cnt} 例重复出现在多个折的 val，已去重（只评一次）")
        # 区分两种"未评估"：
        #  · 所属折**没有权重**（如训练未完成）→ 阶段性正常，只提示；
        #  · 所属折有权重却仍未进任何 val → 划分缺陷，明确告警。
        _val_all = set()
        for f in sorted(folds, key=lambda x: int(x)):
            _val_all |= {x for x in (folds[f].get("val") or []) if x in by_acc}
        _miss_any = sorted(set(by_acc) - _seen)
        _really_lost = sorted(set(by_acc) - _val_all)
        if _miss_any and not _really_lost:
            print(f"[eval-sd] （{len(_miss_any)} 例不在本批已加载折的 val 中——通常是这些折"
                  f"尚无权重，属阶段性正常）")
        elif _really_lost:
            print(f"[eval-sd] ⚠️ {len(_really_lost)} 例**未出现在任何折的 val**，"
                  f"永远拿不到 OOF 预测：{_really_lost[:5]}"
                  f"{' …' if len(_really_lost) > 5 else ''}")
            print("[eval-sd]    （`build_folds` 已修复该缺陷；但重新生成划分会让旧权重"
                  "评到训练过的病例，故当前保留原划分，评估侧不再覆盖这些病例。）")
        if not scopes:
            print("[eval-sd] ⚠️ 无可用折权重，回退为全量模式（--all）")
            scopes = [("all", ckpts, [c["accession"] for c in cases_all])]
    else:
        scopes = [("all", ckpts, [c["accession"] for c in cases_all])]

    if a.oof and folds:
        _scope_txt = f"OOF（逐折评本折 val，{len(scopes)} 折汇总，无泄漏）"
    else:
        _tag = f"，{len(ckpts)} 折集成" if len(ckpts) > 1 else ""
        _scope_txt = f"全量（含训练集⚠，数字会虚高{_tag}）"

    pre = load_config("preprocess.yaml")
    cache_dir = resolve_cache_dir()
    combos_tta = _tta_combos(tuple(pre["inference"].get("tta_flips") or ()))
    tta_batch = int(pre["inference"].get("tta_batch", 2))

    embeds: dict[str, np.ndarray] = {}
    feats: dict[str, dict] = {}
    p_fake: dict[str, float] = {}
    p_stitch: dict[str, float] = {}

    # ---------------- 逐 scope 预测 ----------------
    for tag, ck, accs in scopes:
        if a.limit:
            accs = accs[: a.limit]
        pipe = GliomaPipeline(ck)
        gv = pipe.gcfg or {}
        gsize, gmm = int(gv.get("out", 96)), float(gv.get("size_mm", 192))
        print(f"[eval-sd] scope={tag} 权重{len(ck)}个 {len(accs)} 例", flush=True)

        def _prep(c):
            try:
                vol, _aff, _m = load_case_cached(c, pre, cache_dir)
                return global_view(vol, brain_center(vol), gmm, gsize), None
            except Exception as e:                                # noqa: BLE001
                return None, e

        ex = ThreadPoolExecutor(max_workers=3)
        order = list(range(len(accs)))
        W = 4
        futs = {i: ex.submit(_prep, by_acc[accs[i]]) for i in order[:W]}
        nxt = W
        for i in order:
            while nxt < len(order) and len(futs) < W:
                futs[nxt] = ex.submit(_prep, by_acc[accs[nxt]])
                nxt += 1
            g, err = futs.pop(i).result()
            if err is not None or g is None:
                continue
            acc = accs[i]
            try:
                x = torch.from_numpy(np.ascontiguousarray(g))[None].to(pipe.device, torch.float32)
                r = _forward_batch(pipe.models, x, combos_tta, torch.bfloat16,
                                   tta_batch=tta_batch)
                sp = np.asarray(r["special"].cpu().numpy(), float).ravel()
                embeds[acc] = np.asarray(r["embed"].cpu().numpy(), np.float32)
                p_fake[acc], p_stitch[acc] = float(sp[0]), float(sp[1])
                feats[acc] = case_fingerprint(by_acc[acc])
            except Exception as e:                                # noqa: BLE001
                print(f"[eval-sd] ✗ {acc}: {e}")
            if (i + 1) % 100 == 0:
                print(f"[eval-sd] {tag} {i+1}/{len(accs)} …", flush=True)
        ex.shutdown(wait=False)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 供 scripts/24_verify_eval_split.py 自检读取（不影响正常输出）
    globals()["_LAST_PROBS"] = {"fake": dict(p_fake), "stitch": dict(p_stitch),
                                "n_embeds": len(embeds), "scope": "oof" if (a.oof and folds) else "all"}

    # ---------------- 指标 ----------------
    accs_done = sorted(embeds)
    n_fake = len(pos_fake & set(accs_done))
    n_comp = len(pos_comp & set(accs_done))
    print(f"\n[eval-sd] 评估集={_scope_txt}")
    print(f"[eval-sd] 完成 {len(accs_done)} 例 | 其中 fake {n_fake} | comp {n_comp} | "
          f"gold 对 {sum(1 for x, y in gold if x in embeds and y in embeds)}")

    rep: dict = {"n": len(accs_done), "n_fake_pos": n_fake, "n_comp_pos": n_comp,
                 "scope": "oof" if (a.oof and folds) else "all"}

    if not accs_done:
        print("[eval-sd] ✗ 没有任何病例完成推理")
        return 1

    for kind, pos, pmap in (("fake", pos_fake, p_fake), ("stitch", pos_comp, p_stitch)):
        s = np.array([pmap[x] for x in accs_done])
        y = np.array([1.0 if x in pos else 0.0 for x in accs_done])
        if (y > 0.5).sum() == 0:
            print(f"[eval-sd] ⚠️ {kind}: 评估集内无正样本，跳过 AUC")
            continue
        rep[f"{kind}_auc"] = _auc(s, y)
        rep[f"{kind}_prob_mean_pos"] = float(s[y > 0.5].mean())
        rep[f"{kind}_prob_mean_neg"] = float(s[y <= 0.5].mean()) if (y <= 0.5).any() else None
        rep[f"{kind}_n_pos"] = int((y > 0.5).sum())
        print(f"[eval-sd] {kind}: AUC={rep[f'{kind}_auc']:.4f}  "
              f"正样本均值={rep[f'{kind}_prob_mean_pos']:.4f} "
              f"负样本均值={rep[f'{kind}_prob_mean_neg']}  (n_pos={rep[f'{kind}_n_pos']})")

    gold_in = {g for g in gold if g[0] in embeds and g[1] in embeds}
    if len(gold_in) < 3:
        print(f"[eval-sd] ⚠️ 评估集内重复金标准仅 {len(gold_in)} 对，指标不稳定"
              f"（建议用 OOF 模式以纳入全部 {len(gold)} 对）")
    if gold_in:
        pos, neg = gold_similarities(embeds, [list(g) for g in gold_in])
        calib = calibrate(pos, neg)
        for w in (0.0, a.fp_weight):
            pairs = match_pairs(embeds, calib, topk=int(pre["inference"]["duplicate_topk"]),
                                feats=feats, w_fp=w)
            rep[f"duplicate_fp{w}"] = M.duplicate_report(pairs, gold_in, sorted(embeds))
            r = rep[f"duplicate_fp{w}"]
            print(f"[eval-sd] 重复影像 w_fp={w}: AUC-PR={r['auc_pr']:.5f} "
                  f"Recall@10%FPR={r['recall@10fpr']:.4f} "
                  f"Precision@15%Recall={r['precision@15recall']:.5f} (n_pred={r['n_pred']})",
                  flush=True)
        rep["calib"] = calib
        rep["n_gold_evaluated"] = len(gold_in)

        # 指纹单独区分度（随机采样一批病例两两比较，避免 O(N²) 过慢）
        import random as _rnd
        _rnd.seed(0)
        keys = sorted(embeds)
        sub = sorted(set([x for g in gold_in for x in g]) |
                     set(_rnd.sample(keys, min(80, len(keys)))))
        ps, ns = [], []
        for i, x in enumerate(sub):
            for y2 in sub[i + 1:]:
                fs = fingerprint_similarity(feats.get(x), feats.get(y2))
                if fs is None:
                    continue
                (ps if tuple(sorted((x, y2))) in gold_in else ns).append(fs)
        if ps and ns:
            rep["fingerprint_auc"] = _auc(np.array(ps + ns),
                                          np.array([1.0] * len(ps) + [0.0] * len(ns)))
            rep["fingerprint_gap"] = {"pos_min": float(np.min(ps)), "neg_max": float(np.max(ns))}
            print(f"[eval-sd] 指纹 AUC={rep['fingerprint_auc']:.4f} "
                  f"正对最小={np.min(ps):.4f} 负对最大={np.max(ns):.4f}", flush=True)

    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
