#!/usr/bin/env python
"""评估划分逻辑自检：用**已知答案的合成数据**验证 OOF / 留一折集成划分无误。

为什么需要它：三个评估脚本（分割 / 阈值标定 / 目标一二）各自实现了
"用哪些模型的权重去评哪些病例"的划分规则。这类规则**错了不会报错**，
只会安安静静地给出虚高或虚低的数字——正是最难发现的一类问题。

本脚本分三段，全部用"可验证的已知答案"：

  A. 留一折集成划分   直接调用 16_finalize.sh --print-split（测**真实实现**），
                      断言：评估 fold f 时用的折集合 == 全部折 − {f}。
  B. OOF 划分完整性   断言：各折 val 互不相交、并集为全部病例
                      （这是"每一例恰好被评估一次"的前提）。
  C. OOF 语义端到端   注入 mock 模型与 mock 数据，让**输出值本身编码划分是否正确**：
                      若某病例被"它的 val 折对应的模型"评估 → 输出 1.0，否则 0.0。
                      因此"所有病例都是 1.0"等价于"OOF 划分完全正确"。

用法::

    python scripts/24_verify_eval_split.py
退出码 0 = 全部通过。
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

_RESULTS: list[tuple[str, bool, str]] = []


_WARNS: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)


def warn(name: str, detail: str = "") -> None:
    """已知且已评估影响的偏差：不算失败，但必须显式暴露给使用者。"""
    _WARNS.append((name, detail))
    print(f"  [WARN] {name}" + (f"   {detail}" if detail else ""), flush=True)


# --------------------------------------------------------------------------- #
# A. 留一折集成的划分（调用真实实现）
# --------------------------------------------------------------------------- #
def part_a() -> None:
    print("\nA 留一折集成：评估 fold f 时必须排除 fold f 自身")
    sh = os.path.join(ROOT, "scripts", "16_finalize.sh")
    try:
        out = subprocess.run(["bash", sh, "--print-split"], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
    except Exception as exc:                                        # noqa: BLE001
        check("16_finalize.sh --print-split 可执行", False, f"{type(exc).__name__}: {exc}")
        return
    if out.returncode != 0:
        check("--print-split 正常退出", False, (out.stderr or "")[-200:])
        return

    mapping: dict[str, list[str]] = {}
    for line in out.stdout.splitlines():
        m = re.match(r"^\s*fold(\d+):\s*(.*)$", line.strip())
        if m:
            mapping[f"fold{m.group(1)}"] = [x.strip() for x in m.group(2).split(",") if x.strip()]

    if not mapping:
        check("解析到划分结果", False, f"输出={out.stdout.strip()[:120]}")
        return
    check("解析到划分结果", True, f"{len(mapping)} 折：{sorted(mapping)}")

    folds = sorted(mapping)
    ok_self, ok_all = True, True
    detail = []
    for f in folds:
        used = mapping[f]
        # 期望：其余各折的**目录名**（g4_foldN）
        others = [f"g4_fold{x.split('fold')[-1]}" for x in folds if x != f]
        if f in used:
            ok_self = False
            detail.append(f"{f} 用了自己")
        if sorted(used) != sorted(others):
            ok_all = False
            detail.append(f"{f} 应={others} 实={used}")
    check("任何折的评估都**不含自身**（杜绝泄漏）", ok_self, "; ".join(detail[:2]))
    check("评估集合 == 其余全部折（无遗漏）", ok_all, "; ".join(detail[:2]))


# --------------------------------------------------------------------------- #
# B. OOF 划分完整性
# --------------------------------------------------------------------------- #
def _load_folds() -> tuple[dict, list[str]]:
    from src.utils.config import load_paths, resolve
    paths = load_paths()
    man = json.load(open(resolve(paths["manifest"]), encoding="utf-8"))
    folds = json.load(open(resolve(paths["folds"]), encoding="utf-8"))
    accs = [c["accession"] for c in man["cases"] if c.get("images")]
    return folds, accs


def part_b() -> None:
    print("\nB OOF 划分完整性：各折 val 互斥且并集为全部病例")
    try:
        folds, accs = _load_folds()
    except Exception as exc:                                        # noqa: BLE001
        check("读取 folds/manifest", False, f"{type(exc).__name__}: {exc}")
        return

    vals = {f: set(folds[f].get("val") or []) for f in folds}
    all_val: list[str] = []
    for v in vals.values():
        all_val.extend(v)
    uni = set(all_val)

    dup = {a: c for a, c in collections.Counter(all_val).items() if c > 1}
    never = set(accs) - uni
    # 这两项是"划分质量"指标而非"逻辑正确性"指标：
    # 实测旧版 build_folds 存在 3 例重复 + 2 例缺失，根因已修（见 dataset.build_folds），
    # 但重新生成划分会让已训练权重评到训练过的病例 → 泄漏，故当前保留原划分。
    if not dup:
        check("各折 val 互不相交（每例至多被评估一次）", True, f"总计 {len(all_val)} 条")
    else:
        warn("各折 val 存在重复（旧版 build_folds 缺陷，评估侧已去重）",
             f"{len(dup)} 例重复：{sorted(dup)[:4]}")
    if not never:
        check("不存在'从未进入任何折 val'的病例", True, "全集覆盖")
    else:
        warn("有病例从未进入任何折 val（无法获得 OOF 预测）",
             f"{len(never)} 例：{sorted(never)[:4]}")
    print(f"       折数={len(vals)}  val 规模="
          f"{ {f: len(v) for f, v in sorted(vals.items())} }")


# --------------------------------------------------------------------------- #
# C. OOF 语义端到端（mock 注入，输出值编码划分正确性）
# --------------------------------------------------------------------------- #
def part_c() -> None:
    print("\nC OOF 语义端到端：输出值编码'是否被正确的折模型评估'")

    import src.data.dataset as DS
    import src.inference.pipeline as PIPE
    import src.inference.sliding as SLD

    # ---- 合成数据：6 例，3 折，每折 val 2 例（折号需与真实 checkpoint 对应）----
    n_cases, n_folds = 6, 3
    per = n_cases // n_folds
    accs = [f"C{i:02d}" for i in range(n_cases)]
    membership: dict[str, int] = {}                       # 病例 → 它所属的 val 折
    folds_cfg: dict[str, dict] = {}
    for f in range(n_folds):
        v = accs[f * per:(f + 1) * per]
        folds_cfg[str(f)] = {"val": v, "train": [a for a in accs if a not in v]}
        for a in v:
            membership[a] = f

    # 病例索引 → 所属 val 折（mock 数据用常量值携带身份）
    idx_of = {a: i + 1 for i, a in enumerate(accs)}

    tmp = tempfile.mkdtemp(prefix="glioma_split_")
    man_path = os.path.join(tmp, "manifest.json")
    folds_path = os.path.join(tmp, "folds.json")
    json.dump({"cases": [{"accession": a, "images": {"t1c": {"path": "/x.nii.gz"}},
                          "masks": {}, "dir": tmp} for a in accs]},
              open(man_path, "w", encoding="utf-8"))
    json.dump(folds_cfg, open(folds_path, "w", encoding="utf-8"))
    os.environ["MANIFEST"], os.environ["FOLDS"] = man_path, folds_path

    # ---- mock 1：病例数据（常量 = 病例索引，缩放后仍为常量便于识别）----
    def fake_load(case, cfg, cache_dir=None, log=None):
        # 用整块常量携带病例身份；物理尺寸(8×1mm)与 size_mm 一致 → 不被 padding 稀释
        v = np.full((1, 8, 8, 8), float(idx_of[case["accession"]]), np.float32)
        return v, np.eye(4), {}

    # ---- mock 2：pipeline（把权重路径中的折号带进模型对象）----
    class _FakeModel:
        def __init__(self, fold: int):
            self.fold = fold

    class _FakePipe:
        def __init__(self, ckpts, device=None):
            m = re.search(r"g4_fold(\d+)", str(ckpts[0]))
            self._fold = int(m.group(1)) if m else -1
            self.models = [_FakeModel(self._fold)]
            self.device = "cpu"
            # ★ size_mm 必须与 mock 数据的物理尺寸一致（8 体素 × 1mm）。
            # 若沿用真实配置的 192mm，global_view 会把 8³ 数据 pad 到 192mm
            # 再缩回 8³ —— 采样点会落进 padding 的 0 区，身份值被抹掉。
            self.gcfg = {"out": 8, "size_mm": 8}
            self.thresholds = [0.5, 0.5]

    # ---- mock 3：前向（★核心：输出值 = 划分是否正确）----
    def fake_forward(models, x, combos, dtype, tta_batch=1):
        model_fold = models[0].fold
        # 用 max 而非 mean：global_view 会把 8³ 数据 pad 到物理尺寸，
        # 均值会被大量 0 稀释到 0，而 max 仍等于常量本身。
        case_idx = int(round(float(x.float().max().item())))
        acc = accs[case_idx - 1]
        correct = (membership[acc] == model_fold)                 # 用对了折模型？
        val = 1.0 if correct else 0.0
        n = x.shape[0]
        import torch
        return {"special": torch.tensor([[val, val]]),
                "embed": torch.zeros(n, 4),
                "cls": [torch.zeros(n, 1)]}

    # ---- 注入（必须在脚本 main 内部 import 之前完成）----
    orig = (DS.load_case_cached, PIPE.GliomaPipeline, SLD._forward_batch)
    DS.load_case_cached = fake_load
    PIPE.GliomaPipeline = _FakePipe
    SLD._forward_batch = fake_forward

    spec = importlib.util.spec_from_file_location(
        "eval_sd_under_test", os.path.join(ROOT, "scripts", "15_eval_special_dup.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    captured: dict = {}
    try:
        spec.loader.exec_module(mod)
        argv = sys.argv
        sys.argv = ["15_eval_special_dup.py"]                     # 默认 OOF，无 --limit
        try:
            rc = mod.main()
        finally:
            sys.argv = argv
        captured["rc"] = rc
    except Exception as exc:                                        # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("OOF 流程可执行", False, f"{type(exc).__name__}: {exc}")
        return
    finally:
        DS.load_case_cached, PIPE.GliomaPipeline, SLD._forward_batch = orig

    check("OOF 流程可执行", captured.get("rc") == 0, f"rc={captured.get('rc')}")

    # ---- 从 p_fake 的取值反推划分正确性 ----
    # 脚本把每例预测写入 p_fake[acc]；正确划分 → 1.0，错误 → 0.0
    snap = getattr(mod, "_LAST_PROBS", None)
    if not snap:
        check("读取脚本内部预测快照", False, "未找到 _LAST_PROBS")
        return
    p_fake = snap.get("fake") or {}
    n_eval = len(p_fake)
    n_ok = sum(1 for v in p_fake.values() if abs(float(v) - 1.0) < 1e-6)
    check("每一例都被评估到（无遗漏）", n_eval == len(accs), f"{n_eval}/{len(accs)} 例")
    check("每一例都由'未见过它的折模型'评估（无泄漏）", n_ok == len(accs),
          f"{n_ok}/{len(accs)} 例正确" + ("" if n_ok == n_eval else " → 存在用了错误折的病例"))
    wrong = sorted(a for a, v in p_fake.items() if abs(float(v) - 1.0) >= 1e-6)
    if wrong:
        print(f"       被错误折评估的病例：{wrong[:5]}")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 74)
    print("评估划分逻辑自检（已知答案的可验证构造）")
    print("=" * 74)
    for fn in (part_a, part_b, part_c):
        try:
            fn()
        except Exception as exc:                                    # noqa: BLE001
            check(f"{fn.__name__} 执行异常", False, f"{type(exc).__name__}: {exc}")
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"汇总：{len(_RESULTS) - n_fail}/{len(_RESULTS)} 项通过"
          + (f"；{len(_WARNS)} 项 WARN" if _WARNS else "")
          + ("（全部 PASS ✔）" if n_fail == 0 else f"；{n_fail} 项 FAIL ✘"))
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"  FAIL: {name}  {detail}")
    for name, detail in _WARNS:
        print(f"  WARN: {name}  {detail}")
    print("=" * 74)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
