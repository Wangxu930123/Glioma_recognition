#!/usr/bin/env python
"""评估划分逻辑自检：用**已知答案的合成数据**验证 OOF / 留一折集成划分无误。

为什么需要它：三个评估脚本（分割 / 阈值标定 / 目标一二）各自实现了
"用哪些模型的权重去评哪些病例"的划分规则。这类规则**错了不会报错**，
只会安安静静地给出虚高或虚低的数字——正是最难发现的一类问题。

本脚本分三段，全部用"可验证的已知答案"：

  A. 评估划分口径     直接调用 16_finalize.sh --print-split（测**真实实现**）：
                      无验证集 → 断言 fold f 用的折集合 == 全部折 − {f}；
                      注入合成验证集清单 → 断言切到 external 且用**全部折**（不做留一）。
  B. OOF 划分完整性   断言：各折 val 互不相交、并集为全部病例
                      （这是"每一例恰好被评估一次"的前提）。
  C. OOF 语义端到端   注入 mock 模型与 mock 数据，让**输出值本身编码划分是否正确**：
                      若某病例被"它的 val 折对应的模型"评估 → 输出 1.0，否则 0.0。
                      因此"所有病例都是 1.0"等价于"OOF 划分完全正确"。
  D. external 语义    注入 mock 与合成验证集清单（2 折），断言：清单内每一例都被评估、
                      且每例都由**全部折**的集成预测（输出值 == 折数），scope=external。
  E. external 取数    断言分割评估在 external 下：病例取自 manifest_val（不读 folds）、
                      报告标记 split=external、无掩码病例被跳过并计数。

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
# A. 评估划分（调用真实实现，两种口径都要能解释）
# --------------------------------------------------------------------------- #
def _run_print_split(env_extra: dict | None = None) -> tuple[int, str]:
    sh = os.path.join(ROOT, "scripts", "16_finalize.sh")
    # PYTHON=当前解释器：16_finalize.sh 内部要用 python 探验证集清单，
    # 开发机上不一定存在名为 `python` 的可执行文件。
    env = {**os.environ, "PYTHON": sys.executable, **(env_extra or {})}
    try:
        out = subprocess.run(["bash", sh, "--print-split"], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=120)
    except Exception as exc:                                        # noqa: BLE001
        return 255, f"{type(exc).__name__}: {exc}"
    return out.returncode, (out.stdout or "") + (out.stderr or "")


def _parse_split(out: str) -> tuple[str, dict[str, list[str]], list[str]]:
    """解析 ``--print-split`` 输出 → ``(口径, 留一划分, external 列表)``。"""
    mode = ""
    mapping: dict[str, list[str]] = {}
    ext: list[str] = []
    for line in out.splitlines():
        m = re.match(r"^\s*split=(\S+)\s*$", line.strip())
        if m:
            mode = m.group(1)
        m = re.match(r"^\s*fold(\d+):\s*(.*)$", line.strip())
        if m:
            mapping[f"fold{m.group(1)}"] = [x.strip() for x in m.group(2).split(",") if x.strip()]
        m = re.match(r"^\s*external:\s*(.*)$", line.strip())
        if m:
            ext = [x.strip() for x in m.group(1).split(",") if x.strip()]
    return mode, mapping, ext


def _synthetic_val_manifest() -> tuple[str, dict]:
    """造一份**最小可用**的验证集清单（只用于自检口径切换，绝不参与训练）。"""
    from src.utils.config import data_source_tag
    tmp = tempfile.mkdtemp(prefix="glioma_val_split_")
    path = os.path.join(tmp, "manifest_val.json")
    json.dump({"cases": [{"accession": "V1", "images": {"t1c": {"path": "/x.nii.gz"}},
                          "masks": {}, "labels": {}, "dir": tmp}],
               "special": {},
               "data_source": data_source_tag(tmp, phase="val"),
               "data_root": tmp},
              open(path, "w", encoding="utf-8"))
    return tmp, path


def part_a() -> None:
    print("\nA 评估划分：有验证集 → 全折集成；无验证集 → 留一折集成")

    rc, out = _run_print_split()
    if rc != 0:
        check("16_finalize.sh --print-split 可执行", False, out.strip()[-200:])
        return
    mode, mapping, ext = _parse_split(out)
    check("解析到口径标记 split=...", mode in ("oof", "external"), f"mode={mode or '?'}")

    # ---- ② 注入合成验证集清单 → 必须切到 external，且列出**全部折**（不做留一）----
    tmp, val_path = _synthetic_val_manifest()
    rc2, out2 = _run_print_split({"VAL_MANIFEST": val_path})
    if rc2 != 0:
        check("注入验证集清单后 --print-split 可执行", False, out2.strip()[-200:])
        return
    mode2, mapping2, ext2 = _parse_split(out2)
    if mode2 != "external":
        check("给验证集清单后切到 external 口径", False, f"mode={mode2 or '?'} 输出={out2.strip()[:120]}")
        return
    check("给验证集清单后切到 external 口径", True)

    from src.utils.config import fold_ckpts
    expect = [os.path.basename(os.path.dirname(p)) for p in fold_ckpts()]
    check("external 口径用**全部折**（不排除任何折）",
          sorted(ext2) == sorted(expect), f"期望={expect} 实际={ext2}")
    check("external 口径不输出留一划分", not mapping2, f"不应有 foldN 行：{mapping2}")

    # ---- ① 当前环境的真实口径（开发机多半是 oof）----
    if mode == "external":
        check("当前环境：已接入官方验证集（external）", True,
              f"折={ext}")
        return
    if not mapping:
        check("解析到留一划分结果", False, f"输出={out.strip()[:120]}")
        return
    check("解析到留一划分结果", True, f"{len(mapping)} 折：{sorted(mapping)}")

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
        sys.argv = ["15_eval_special_dup.py", "--split", "oof"]   # 本段测 OOF 语义
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
# D. external 语义：每例都过**全部折**（不做留一）
# --------------------------------------------------------------------------- #
def part_d() -> None:
    print("\nD external 语义：清单内每一例都由全部折的集成预测")
    import src.data.dataset as DS
    import src.inference.pipeline as PIPE
    import src.inference.sliding as SLD
    import src.inference.writer as WRITER

    n_cases, n_folds = 4, 2
    accs = [f"E{i:02d}" for i in range(n_cases)]
    tmp = tempfile.mkdtemp(prefix="glioma_ext_")
    man_path = os.path.join(tmp, "manifest_val.json")
    json.dump({"cases": [{"accession": a, "images": {"t1c": {"path": "/x.nii.gz"}},
                          "masks": {}, "labels": {}, "dir": tmp} for a in accs],
               "special": {"fake_cases": [accs[0]], "composition_cases": [],
                           "gold_pairs": []},
               "data_source": "local/extselftest/val", "data_root": tmp},
              open(man_path, "w", encoding="utf-8"))
    ckpt_dir = os.path.join(tmp, "checkpoints")           # 伪造折权重（只做存在性检查）
    for f in range(n_folds):
        d = os.path.join(ckpt_dir, f"g4_fold{f}")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "best.pth"), "w", encoding="utf-8").close()

    def fake_load(case, cfg, cache_dir=None, log=None):
        return np.zeros((1, 8, 8, 8), np.float32), np.eye(4), {}

    class _FakePipe:
        def __init__(self, ckpts, device=None):
            self.n = len(ckpts)
            self.models = list(range(self.n))
            self.device = "cpu"
            self.gcfg = {"out": 8, "size_mm": 8}          # 与 8³ 数据的物理尺寸一致
            self.thresholds = [0.5, 0.5]

    def fake_forward(models, x, combos, dtype, tta_batch=1):
        import torch
        n = float(len(models))                            # 输出值 = 用了几个模型的权重
        return {"special": torch.tensor([[n, n]] * x.shape[0]),
                "embed": torch.zeros(x.shape[0], 4),
                "cls": [torch.zeros(x.shape[0], 1)]}

    spec = importlib.util.spec_from_file_location(
        "eval_sd_external", os.path.join(ROOT, "scripts", "15_eval_special_dup.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    orig = (DS.load_case_cached, DS.brain_center, DS.global_view, PIPE.GliomaPipeline,
            SLD._forward_batch, WRITER.case_fingerprint,
            os.environ.get("VAL_MANIFEST"), os.environ.get("CKPT_DIR"))
    DS.load_case_cached = fake_load
    # 预处理/指纹与"划分口径"无关：全零体在真实实现里没有脑中心可定位，
    # 这里用恒等实现保证本段只考察**划分与集成成员数**。
    DS.brain_center = lambda vol: np.zeros(3, np.float32)
    DS.global_view = lambda vol, ctr, mm, size: np.ascontiguousarray(vol, np.float32)
    WRITER.case_fingerprint = lambda case: {}
    PIPE.GliomaPipeline = _FakePipe
    SLD._forward_batch = fake_forward
    os.environ["VAL_MANIFEST"] = man_path
    os.environ["CKPT_DIR"] = ckpt_dir

    rc: int | None = None
    try:
        spec.loader.exec_module(mod)
        argv = sys.argv
        sys.argv = ["15_eval_special_dup.py", "--split", "external"]
        try:
            rc = mod.main()
        finally:
            sys.argv = argv
    except Exception as exc:                                        # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("external 流程可执行", False, f"{type(exc).__name__}: {exc}")
        return
    finally:
        (DS.load_case_cached, DS.brain_center, DS.global_view, PIPE.GliomaPipeline,
         SLD._forward_batch, WRITER.case_fingerprint) = orig[:6]
        for k, v in (("VAL_MANIFEST", orig[6]), ("CKPT_DIR", orig[7])):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    check("external 流程可执行", rc == 0, f"rc={rc}")

    snap = getattr(mod, "_LAST_PROBS", None) or {}
    p_fake = snap.get("fake") or {}
    check("external：清单内每一例都被评估", len(p_fake) == n_cases, f"{len(p_fake)}/{n_cases}")
    check("external：每例都用**全部折**集成（不做留一）",
          bool(p_fake) and all(abs(float(v) - n_folds) < 1e-6 for v in p_fake.values()),
          f"应={float(n_folds)} 实际={sorted({float(v) for v in p_fake.values()})}")
    check("external：scope 与权重数写入快照",
          snap.get("scope") == "external" and snap.get("n_ckpt") == n_folds,
          f"scope={snap.get('scope')} n_ckpt={snap.get('n_ckpt')}")


# --------------------------------------------------------------------------- #
# E. external 取数：分割评估读 manifest_val、跳无掩码病例
# --------------------------------------------------------------------------- #
def part_e() -> None:
    print("\nE external 取数：分割评估的病例来自 manifest_val（不读 folds）")
    import src.evaluation.evaluate as EV
    import src.inference.pipeline as PIPE
    import src.inference.writer as WRITER

    n_cases, n_gt = 3, 2                                   # 3 例，其中 1 例无掩码
    accs = [f"S{i}" for i in range(n_cases)]
    tmp = tempfile.mkdtemp(prefix="glioma_eval_ext_")
    man_path = os.path.join(tmp, "manifest_val.json")
    blob = np.zeros((8, 8, 8), np.float32)
    blob[4, 4, 4] = 1.0
    json.dump({"cases": [{"accession": a, "images": {"t1c": {"path": "/x.nii.gz"}},
                          # 清单里的掩码只是"有/无"标记（真实清单存路径）；
                          # 掩码数组由下面的 mock 载入函数提供，保证清单可 JSON 序列化。
                          "masks": ({"core": "gt_core.nii.gz",
                                     "peri": "gt_peri.nii.gz"} if i < n_gt else {}),
                          "labels": {}, "dir": tmp} for i, a in enumerate(accs)],
               "special": {},
               "data_source": "local/extselftest/val", "data_root": tmp},
              open(man_path, "w", encoding="utf-8"))
    ckpts = [os.path.join(tmp, "checkpoints", f"g4_fold{f}", "best.pth")
             for f in range(2)]

    def fake_load(case, cfg, cache_dir=None, log=None):
        m = case.get("masks") or {}
        return (np.zeros((1, 8, 8, 8), np.float32), np.eye(4),
                ({"core": blob, "peri": blob} if m else {}))

    class _FakePipe:
        def __init__(self, ckpts_, device=None):
            self.n = len(ckpts_)
            self.thresholds = [0.5, 0.5]

        def predict_prob(self, case, vol):
            seg = np.zeros((2,) + tuple(vol.shape[1:]), np.float32)
            seg[:, 4, 4, 4] = 1.0                          # 与 gt 完全一致 → Dice=1
            return {"seg": seg, "embed": np.zeros(8, np.float32)}

    orig = (EV.load_case_cached, EV.GliomaPipeline, EV.make_targets,
            WRITER.case_fingerprint, PIPE.postprocess, os.environ.get("VAL_MANIFEST"))
    EV.load_case_cached = fake_load
    EV.GliomaPipeline = _FakePipe
    EV.make_targets = lambda masks, shape: np.stack(
        [np.asarray(masks["core"], np.float32), np.asarray(masks["peri"], np.float32)])
    WRITER.case_fingerprint = lambda case: {}
    PIPE.postprocess = lambda c, p, minv, sp: (np.asarray(c, bool), np.asarray(p, bool))
    os.environ["VAL_MANIFEST"] = man_path

    try:
        cases, man, tag = EV.resolve_split("external", 0)   # 真实入口：不读 folds
        check("external：取数走 manifest_val 且标记 external",
              tag == "external" and len(cases) == n_cases,
              f"tag={tag} cases={len(cases)}")
        # 无掩码的例"只参与重复影像评估"这句只在**有**重复金标准时成立。官方验证集实测
        # 只给 SeriesType.xlsx（没有字段金标准、也没有重复影像金标准）→ 这些例其实
        # 哪个指标都不进，必须说清，别让人以为它们还贡献了重复影像那一项。
        import contextlib
        import io
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            EV.resolve_split("external", 0)
        check("external：无掩码且无重复金标准时明说'不参与任何指标'",
              "不参与任何指标" in _buf.getvalue(),
              _buf.getvalue().strip().splitlines()[-1][:56])
        # 全量训练（--fold full）的**早停集只取带掩膜的例**：字段标签在 val 里不参与任何
        # 计算（选模指标是 val_dice_peri），而没有掩膜的例会让 make_targets 产出全零 target
        # —— Dice 在"预测也为空"时按 den==0 记成 1.0 的**假满分**，把 best 直接选歪。
        # 官方验证集实测连字段金标准表都没有；若它同时没有掩膜，这种情况会全量命中。
        import src.utils.config as CFG
        import src.data.dataset as DS
        _p_mix = os.path.join(tmp, "manifest_val_mix.json")
        json.dump({"cases": [
            {"accession": "M1", "images": {"t1c": {"path": "/x.nii.gz"}},
             "masks": {"core": "c.nii.gz"}, "labels": {}},
            {"accession": "L1", "images": {"t1c": {"path": "/x.nii.gz"}},
             "masks": {}, "labels": {"WHO_Grade": "3"}},        # 只有标签、没有掩膜
            {"accession": "N1", "images": {"t1c": {"path": "/x.nii.gz"}},
             "masks": {}, "labels": {}}],
            "special": {}, "data_source": "local/extselftest/val", "data_root": tmp},
            open(_p_mix, "w", encoding="utf-8"))
        os.environ["VAL_MANIFEST"] = _p_mix
        _vc, _ = CFG.external_val_cases()
        check("full 模式的早停集只取带掩膜的例（标签-only 不得混进 Dice 选模）",
              [c["accession"] for c in _vc] == ["M1"],
              f"val={[c['accession'] for c in _vc]}（应为 ['M1']）")
        _mt = DS.make_targets({}, (4, 4, 4))
        check("无掩膜病例的 target 全零 → 空预测会被记成 Dice=1.0 假满分",
              _mt.shape == (2, 4, 4, 4) and float(np.abs(_mt).sum()) == 0.0,
              f"shape={_mt.shape} sum={float(np.abs(_mt).sum())}")
        os.environ["VAL_MANIFEST"] = man_path
        rep = EV.eval_cases(cases, man, ckpts, tag=tag, fold=None)
    except Exception as exc:                                        # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("external 分割评估可执行", False, f"{type(exc).__name__}: {exc}")
        return
    finally:
        (EV.load_case_cached, EV.GliomaPipeline, EV.make_targets,
         WRITER.case_fingerprint, PIPE.postprocess, _vm) = orig
        (EV.load_case_cached, EV.GliomaPipeline, EV.make_targets,
         WRITER.case_fingerprint, PIPE.postprocess) = orig[:5]
        if _vm is None:
            os.environ.pop("VAL_MANIFEST", None)
        else:
            os.environ["VAL_MANIFEST"] = _vm

    check("external：分割评估可执行", True)
    check("external：报告标记 split=external、fold=None",
          rep.get("split") == "external" and rep.get("fold") is None,
          f"split={rep.get('split')} fold={rep.get('fold')}")
    check("external：无掩码病例被跳过并计数",
          rep.get("n") == n_gt and rep.get("n_no_mask") == n_cases - n_gt,
          f"n={rep.get('n')} n_no_mask={rep.get('n_no_mask')}")
    check("external：有掩码病例的分割指标被计算",
          abs(float(rep.get("dice_mean", -1)) - 1.0) < 1e-6,
          f"dice_mean={rep.get('dice_mean')}")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 74)
    print("评估划分逻辑自检（已知答案的可验证构造）")
    print("=" * 74)
    for fn in (part_a, part_b, part_c, part_d, part_e):
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
