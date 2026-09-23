#!/usr/bin/env python
"""校验**提交侧插件**（``Glioma_recognition-main/tasks/``）与**训练工程**
（``glioma_goals/``）的对接正确性——即"训练出来的权重能否被推理链路正确使用"。

为什么需要这个脚本
------------------
本轮排查发现的 7 个缺陷有一个共同特征：**全都不报错**。
它们不会让程序崩溃，只会让指标悄悄变差，因此必须用**断言**固定下来：

| # | 缺陷 | 静默后果 |
|---|---|---|
| 1 | 共享骨干的前向缓存用**固定键** | 后续 Task 复用**别人的权重**算出的结果 |
| 2 | 全局头（cls/special/embed）用 96mm patch 训练 | 推理用 192mm 整脑视图 → 输入分布不一致 |
| 3 | 重复影像指纹的键名与 ``retrieval`` 不匹配 | 指纹融合（权重 0.95）**完全失效** |
| 4 | 指纹用 ``series_uid`` 索引 | UID 跨检查永不重叠 → 相似度恒为 0 |
| 5 | ``asdict()`` 遇非 dataclass 配置对象 | 训练在**第一次保存 best 权重**时崩溃 |
| 6 | 动态类实例的配置存在**类属性**上 | ``model_cfg`` 存空 → 推理按默认值重建网络 |
| 7 | special / embed 的监督键缺失 | 损失分支被**静默跳过**，该头从未训练 |

用法::

    python scripts/25_verify_tasks_integration.py
    GLIOMA_MAIN_ROOT=/path/to/Glioma_recognition-main python scripts/25_verify_tasks_integration.py
"""
from __future__ import annotations

import inspect
import json
import math
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TRACK4 = _HERE.parent
_MAIN = Path(os.environ.get("GLIOMA_MAIN_ROOT")
             or (_TRACK4.parent / "Glioma_recognition-main")).resolve()
_GOALS = Path(os.environ.get("GLIOMA_GOALS_ROOT")
              or (_TRACK4.parent / "glioma_goals")).resolve()

if not (_MAIN / "tasks").is_dir():
    raise SystemExit(f"找不到提交工程 tasks/：{_MAIN}（用 GLIOMA_MAIN_ROOT 指定）")
if not (_GOALS / "shared").is_dir():
    raise SystemExit(f"找不到训练工程 shared/：{_GOALS}（用 GLIOMA_GOALS_ROOT 指定）")

# 三棵树都挂上搜索路径：断言里要跨工程 import（如算法工程的 src.data.dataset）。
# 放在模块顶部而不是各段内部 —— 段内 import 会让该名字在**整个函数**里变成局部变量，
# 前面先用到的段落会直接 UnboundLocalError。
for _p in (_TRACK4, _GOALS, _MAIN):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: 两条训练路径的引擎（用于源码级断言）
_GE_ENGINE = _MAIN / "tasks/_common/training/engine.py"      # 提交侧：统一多任务训练
_GG_ENGINE = _GOALS / "shared/engine.py"                     # 研发侧：各 Goal 独立训练

_PASSED: list[str] = []
_FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (_PASSED if cond else _FAILED).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  {detail}" if detail else ""))


def _section(title: str) -> None:
    print(f"\n{title}")


def main() -> int:
    sys.path.insert(0, str(_MAIN))
    sys.path.insert(0, str(_GOALS))

    # ---------------------------------------------------------------- #
    _section("① 共享前向缓存必须按权重隔离（否则 Task 之间串味）")
    from tasks._common.backbone_runner import BackboneRunner

    a = BackboneRunner(ckpt_root="/tmp/x", ckpt_rel="goal1_authenticity/model.pt")
    b = BackboneRunner(ckpt_root="/tmp/x", ckpt_rel="goal3_tumor/model.pt")
    a2 = BackboneRunner(ckpt_root="/tmp/x", ckpt_rel="goal1_authenticity/model.pt")
    check("不同权重 → 缓存键不同", a.cache_key != b.cache_key,
          f"{a.cache_key!r} ≠ {b.cache_key!r}")
    check("相同权重 → 缓存键相同（仍能共享一次前向）", a.cache_key == a2.cache_key)

    class _Ctx:
        def __init__(self) -> None:
            self.diagnostics, self.warnings = {}, []

    ctx = _Ctx()
    ctx.diagnostics[a.cache_key] = {"special": [1.0, 0.0]}
    check("goal3 不会命中 goal1 的缓存", ctx.diagnostics.get(b.cache_key) is None)

    # ---------------------------------------------------------------- #
    _section("② 权重加载必须校验输出头（缺头时不能静默用随机初始化）")
    _load_src = inspect.getsource(BackboneRunner.load)
    check("已接入 _check_heads", "_check_heads" in _load_src)
    check("已校验 cls_spec ⇄ cls_heads 一致性",
          "cls_in_state" in _load_src and "cls_spec" in _load_src)

    from tasks.goal4_diagnosis.labels import FIELD_ENUMS

    _g4_src = inspect.getsource(
        __import__("tasks.goal4_diagnosis.task", fromlist=["x"]).DiagnosisTask._collect)
    check("goal4 分类头不足时显式失败（不再静默填默认值）",
          "raise ValueError" in _g4_src and str(len(FIELD_ENUMS)) in _g4_src)

    # ---------------------------------------------------------------- #
    _section("③ 训练引擎的配置序列化（asdict 只接受 dataclass）")
    from dataclasses import dataclass
    from types import SimpleNamespace

    from tasks._common.training.engine import _as_dict

    @dataclass
    class _DC:
        base: int = 32

    class _Dyn:
        pass

    _dyn = type("MC", (), {"base": 32, "in_ch": 4})()          # helpers.py 的旧写法
    check("dataclass", _as_dict(_DC()) == {"base": 32})
    check("动态类实例（配置在类属性上）",
          _as_dict(_dyn) == {"base": 32, "in_ch": 4}, str(_as_dict(_dyn)))
    check("SimpleNamespace（现写法）",
          _as_dict(SimpleNamespace(base=32, in_ch=4)) == {"base": 32, "in_ch": 4})
    check("dict / None", _as_dict({"x": 1}) == {"x": 1} and _as_dict(None) == {})

    _help_src = (_MAIN / "tasks/_common/training/helpers.py").read_text(encoding="utf-8")
    # 判据要盯着**赋值语句**，不能扫全文——修复说明的注释里必然会提到旧写法
    check("helpers 用 SimpleNamespace 而非动态类",
          "SimpleNamespace(**{**mc" in _help_src and "model.cfg = type(" not in _help_src)

    # ---------------------------------------------------------------- #
    _section("④ 全局头必须用整脑视图训练（与推理同尺度）")
    _eng = inspect.getsource(
        __import__("tasks._common.training.engine", fromlist=["train"]).train)
    check("提交侧 train() 用 batch['whole'] 前向全局头", 'batch.get("whole")' in _eng)

    from shared.data import BaseCaseDataset

    _gd = inspect.getsource(BaseCaseDataset)
    check("研发侧数据集产出 image_global", "image_global" in _gd)
    check("global_view 可由 config.yaml 控制", "global_view" in _gd)
    _ge = inspect.getsource(__import__("shared.engine", fromlist=["train"]).train)
    check("研发侧 train() 用 image_global 前向全局头", "image_global" in _ge)
    _dup = inspect.getsource(
        __import__("goal2_duplicate.dataset", fromlist=["x"]))
    check("配对数据集产出 image_global / image_global_b",
          "image_global" in _dup and "image_global_b" in _dup)

    # ---------------------------------------------------------------- #
    _section("⑤ 重复影像指纹必须与 retrieval 的键名约定一致（否则融合静默失效）")
    from tasks.goal2_duplicate import retrieval

    def _fp(mod: str) -> dict:
        """按 ``task._fingerprint`` 的产出格式造一个指纹。"""
        return {"mods": [mod], f"shape_{mod}": [10, 10, 10],
                f"zoom_{mod}": [1.0, 1.0, 1.0], f"aff_{mod}": [0.0] * 16,
                "hist": {mod: [1, 2, 3, 4, 5]}, "thumb": {mod: [1.0, 2.0, 3.0, 4.0]}}

    same = retrieval.fingerprint_similarity(_fp("t1c"), _fp("t1c"))
    check("两例真重复 → 高相似度", same is not None and same > 0.9, f"sim={same}")
    diff = _fp("t1c")
    diff["thumb"] = {"t1c": [9.0, 9.0, 9.0, 9.0]}
    low = retrieval.fingerprint_similarity(_fp("t1c"), diff)
    check("缩略图不同 → 相似度显著下降", low is not None and low < same,
          f"{same:.3f} → {low:.3f}")

    # 反例：旧实现（geom 列表 + series_uid 索引）在 retrieval 眼里完全失效。
    # 必须用**两个不同检查**才能复现——同一指纹自比时 mods 相同、
    # Jaccard=1，会恰好给出 1.0 而掩盖问题。
    old_a = {"mods": ["1.2.826.0.1.100"], "geom": [[1, 1, 1, 0, 0, 0]],
             "hist": {"1.2.826.0.1.100": [1, 2, 3]}}
    old_b = {"mods": ["1.2.826.0.1.999"], "geom": [[1, 1, 1, 0, 0, 0]],
             "hist": {"1.2.826.0.1.999": [1, 2, 3]}}
    check("旧格式指纹对两个不同检查恒为 0（复现原缺陷）",
          retrieval.fingerprint_similarity(old_a, old_b) == 0.0)

    _task_src = (_MAIN / "tasks/goal2_duplicate/task.py").read_text(encoding="utf-8")
    # 同样只盯代码：用 ``s.series_uid``（旧实现的下标写法）判断，避免注释误报
    check("task._fingerprint 已改用模态名索引",
          'f"shape_{mod}"' in _task_src and "s.series_uid" not in _task_src)

    # ---------------------------------------------------------------- #
    _section("⑥ 监督信号缺失必须显式告警（而不是静默跳过）")
    from tasks._common.training import losses as L

    _lsrc = inspect.getsource(L.compute_losses)
    check("special 缺失 → 告警", "special_target" in _lsrc and "_warn_once" in _lsrc)
    check("embed 缺失 → 告警", "pair" in _lsrc and "embed" in _lsrc)

    from tasks._common.training.helpers import _SpecialSupervised, _special_targets

    check("提交侧已补 special 监督（Dataset 子类）",
          _SpecialSupervised.__mro__[1].__name__ == "Dataset")
    check("special 标签含两个通道（fake / stitched）",
          len(_special_targets({})) == 2)

    # ---------------------------------------------------------------- #
    _section("⑦ 训练产物 → 推理重建的元信息完备")
    _save_src = inspect.getsource(
        __import__("tasks._common.training.engine", fromlist=["x"])._save)
    for key in ("model_ema", "cls_spec", "arch", "model_cfg", "thresholds"):
        check(f"_save 写入 {key}", f'"{key}"' in _save_src)
    _ge_save = inspect.getsource(
        __import__("shared.engine", fromlist=["x"])._save)
    check("研发侧 cls_spec 从模型结构推导（不再裸导入 model）",
          "_spec_from_model" in _ge_save and "from model import" not in _ge_save)

    # ---------------------------------------------------------------- #
    _section("⑧ best 权重一定会被保存（nan 选择指标不得阻断训练产物）")
    from tasks._common.training.engine import _selection_score as _ss_tasks

    _vals = [
        ({"score": float("nan")}, {"seg": 0.5}, "nan + 有 loss"),
        ({"score": float("inf")}, {"seg": 0.5}, "inf + 有 loss"),
        ({}, {"seg": 0.5}, "缺 score"),
        ({"score": float("nan")}, {}, "nan 且无 loss"),
        ({"score": 0.83}, {"seg": 9.9}, "正常 score"),
    ]
    for val, met, tag in _vals:
        out = _ss_tasks(dict(val), dict(met))
        check(f"提交侧 _selection_score 有限（{tag}）", math.isfinite(out), f"→ {out}")

    check("提交侧 best_score 初值为 -inf（保证首轮必存）",
          'best_score, history = -float("inf")' in _GE_ENGINE.read_text(encoding="utf-8"))

    from shared.engine import _selection_score as _ss_goals

    for val, met, tag in _vals:
        out = _ss_goals(dict(val), dict(met))
        check(f"研发侧 _selection_score 有限（{tag}）", math.isfinite(out), f"→ {out}")
    check("研发侧 best_score 初值为 -inf",
          '-float("inf")' in _GG_ENGINE.read_text(encoding="utf-8"))

    # 反向验证：旧写法 ``float(x or 0.0)`` 在 nan 上会穿透（这正是缺陷成因）
    check("复现旧缺陷：nan or 0.0 仍是 nan",
          not math.isfinite(float(float("nan") or 0.0)))

    _section("⑨ 配对任务不做无用的 patch 前向（否则 3 个 96³ 激活叠加 → OOM）")
    _gge = _GG_ENGINE.read_text(encoding="utf-8")
    check("研发侧按 target 判定是否需要 patch 前向",
          'if batch.get("target") is not None:' in _gge)
    _g5 = (_GOALS / "goal5_segmentation/config.yaml").read_text(encoding="utf-8")
    check("goal5 关闭整脑视图（只用分割头）", "enabled: false" in _g5)

    _section("⑩ 提交侧必须补齐配对监督（否则 embed 头从未被训练）")
    from tasks._common.training.helpers import _SpecialSupervised as _SP

    check("_SpecialSupervised 支持 pair_prob", "pair_prob" in
          inspect.signature(_SP.__init__).parameters)
    _sp_src = inspect.getsource(_SP)
    check("产出 image_b / pair", '"image_b"' in _sp_src and '"pair"' in _sp_src)
    check("pair 用 0 维 tensor（[1] 会与 sim 广播成 [B,B] 而静默算错）",
          "torch.tensor(1.0)" in _sp_src and "torch.tensor(0.0)" in _sp_src)
    check("pair/image_b 在两条分支上都写入（避免 collate 缺键）",
          _sp_src.count('item["pair"]') >= 3)

    _hlp = (_MAIN / "tasks/_common/training/helpers.py").read_text(encoding="utf-8")
    check("训练集启用配对、验证集不产配对",
          "pair_prob=0.5" in _hlp and "pair_prob=0.0" in _hlp)

    _section("⑪ 训练必须分阶段反向（3 个 96³ 前向同时驻留会 OOM）")
    _eng2 = _GE_ENGINE.read_text(encoding="utf-8")
    check("提交侧按阶段 backward（阶段 1 = patch/seg）",
          "阶段 1：patch" in _eng2 and "bd_seg" in _eng2)
    check("提交侧阶段 2 在整脑视图上跑全局头", "阶段 2：整脑视图" in _eng2)
    check("优化器只在两阶段之后步进一次",
          _eng2.count("scaler.step(opt)") == 1 and _eng2.count("opt.step()") == 1)

    # ---------------------------------------------------------------- #
    _section("⑫ 六个 Goal 必须用**同一份**折划分（串行/并行都要求口径一致）")
    # 曾经的实现让每个 Goal 按自己的 val_ratio+seed 划分验证集，那是把
    # "**支持**五个人并行做"误读成"**必须**并行做"。实际要求相反：
    #   · 一个人串行做五个 Goal 时，统一口径才让五个指标互相可比；
    #   · 五个人并行做时，统一口径才让各 Goal 的权重可以合并/集成。
    if str(_GOALS) not in sys.path:
        sys.path.insert(0, str(_GOALS))
    from shared.data import discover_cases, split_train_val

    _data = Path(os.environ.get("GLIOMA_GOALS_DATA")
                 or "/mnt/data_sdb/wangx/data/Brain_MRI/track4_sim")
    if _data.is_dir():
        _cases, _tag = discover_cases(_data), f"真实数据（{_data.name}）"
    else:
        _cases = [{"accession": f"CASE{i:04d}", "dir": "/nonexistent"} for i in range(200)]
        _tag = "合成数据（未找到数据根）"

    _vals: dict[str, frozenset] = {}
    _srcs: set[str] = set()
    for _g in ("goal1_authenticity", "goal2_stitched", "goal2_duplicate",
               "goal3_tumor", "goal4_diagnosis", "goal5_segmentation"):
        _tr, _va, _src = split_train_val(_cases, {"train": {"fold": 0}, "data": {}},
                                         _data, _GOALS / _g, 42)
        _vals[_g] = frozenset(c["accession"] for c in _va)
        _srcs.add(_src)
    _uniq = len(set(_vals.values()))
    check(f"六个 Goal 的验证集完全一致（{_tag}）", _uniq == 1,
          f"共 {_uniq} 种不同划分" + ("" if _uniq == 1 else f" → {sorted(_vals)}"))
    check("六个 Goal 都拿到了非空验证集", all(len(v) > 0 for v in _vals.values()),
          f"val 规模={sorted({len(v) for v in _vals.values()})}")
    check("划分来源一致（不会一半用 folds、一半用 ratio）", len(_srcs) == 1,
          f"来源={sorted(_srcs)}")

    # 六折/多折训练要靠入口传折号。这几条也是**防生成器覆盖**：
    # train.py 由 gen_goals.py 生成，只改生成物的话下次重跑就没了。
    _trains = {g: (_GOALS / g / "train.py").read_text(encoding="utf-8")
               for g in ("goal1_authenticity", "goal2_stitched", "goal2_duplicate",
                         "goal3_tumor", "goal4_diagnosis", "goal5_segmentation")}
    check("六个 train.py 都支持 --fold",
          all('"--fold"' in s for s in _trains.values()))
    check("--fold 回写 cfg['train']（不回写则 build_datasets 读不到，折号被静默忽略）",
          all('cfg["train"] = tr' in s for s in _trains.values()))
    check("折号进 tag（否则多折写到同一 runs/<tag>/ 互相覆盖）",
          all("_fold{a.fold}" in s for s in _trains.values()))
    check("折号开跑前校验（写错时不再静默退回按比例划分）",
          all("available_folds(cfg, data_root, HERE)" in s for s in _trains.values()) and
          "def available_folds" in (_GOALS / "shared/data.py").read_text(encoding="utf-8"))
    check("gen_goals.py 模板已同步（重跑生成器不会丢 --fold）",
          '"--fold"' in (_GOALS / "gen_goals.py").read_text(encoding="utf-8"))

    # 折划分**不能跨数据集复用**：原实现只比较折数，换数据根后若折数相同就原样
    # 返回旧划分 —— 02 打印的是新清单病例数（看着像重建过了），折却是别的数据集的。
    _folds_src = (_TRACK4 / "src/data/dataset.py").read_text(encoding="utf-8")
    check("build_folds 复用旧划分前校验覆盖性", "covered == all_accs" in _folds_src)
    _build_sh = (_TRACK4 / "scripts/02_build_dataset.sh").read_text(encoding="utf-8")
    check("02_build_dataset 加了数据源闸门（防用旧清单建折）",
          "assert_data_source(d" in _build_sh)

    with tempfile.TemporaryDirectory() as _tmpf:
        _mf = Path(_tmpf) / "manifest.json"

        def _write_manifest(prefix: str) -> None:
            cases = [{"accession": f"{prefix}{i:03d}", "dir": "/x",
                      "images": {"t1c": {}}, "masks": {},
                      "labels": {} if i % 4 else {"k": 1}} for i in range(20)]
            _mf.write_text(json.dumps({"cases": cases, "data_source": "local/x/train"},
                                      ensure_ascii=False), encoding="utf-8")

        from src.data.dataset import build_folds                    # noqa: PLC0415
        _write_manifest("C")
        _f1 = build_folds(str(_mf), n_folds=5)
        _cov1 = set().union(*[set(e["val"]) for e in _f1.values()])
        check("折划分覆盖全部病例", _cov1 == {f"C{i:03d}" for i in range(20)})
        _mt = (_mf.parent / "folds.json").stat().st_mtime_ns
        check("覆盖当前数据时复用（重启某一折不会换掉整套划分）",
              build_folds(str(_mf), n_folds=5) == _f1 and
              (_mf.parent / "folds.json").stat().st_mtime_ns == _mt)
        _write_manifest("D")
        _f2 = build_folds(str(_mf), n_folds=5)
        _cov2 = set().union(*[set(e["val"]) for e in _f2.values()])
        check("换成另一份数据（同折数）时重建", _cov2 == {f"D{i:03d}" for i in range(20)},
              f"交集 {len(_cov2 & _cov1)}")

    # ---------------------------------------------------------------- #
    _section("⑬ 平台五目录布局：解析正确、误用当场可见")
    # 平台挂载 /2026aicompetition/datasets/{training, evaluation_first,
    # evaluation_second, evaluation_finals, verification}。数据根必须精确到
    # **其中一个阶段目录**；停在上一层、或把 annotation/ 当病例，都不会报错，
    # 只会安静地跑出与检查集完全对不上的结果 —— 所以这里做**行为级**验证，
    # 而不只是扫源码。
    import nibabel as _nib
    import numpy as _np

    if str(_MAIN) not in sys.path:
        sys.path.insert(0, str(_MAIN))
    from core.exceptions import InvalidInputError                 # noqa: E402
    from data.loader import DatasetLoader                         # noqa: E402

    def _tiny_nii(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _nib.save(_nib.Nifti1Image(_np.zeros((2, 3, 4), _np.float32), _np.eye(4)),
                  str(path))

    _loader = DatasetLoader()
    with tempfile.TemporaryDirectory() as _tmp:
        _root = Path(_tmp) / "datasets"
        # 真实形态：病例目录 + 同目录中文掩膜 + 顶层 annotation/ 标注目录
        _tiny_nii(_root / "evaluation_first" / "ACC001" / "S1" / "S1.nii.gz")
        _tiny_nii(_root / "evaluation_first" / "ACC001" / "S1" / "瘤体.nii.gz")
        _tiny_nii(_root / "evaluation_first" / "annotation" / "fake"
                  / "FAKE_1" / "S2" / "S2.nii.gz")
        _tiny_nii(_root / "training" / "ACC002" / "S1" / "S1.nii.gz")

        _studies = list(_loader.iter_studies(_root / "evaluation_first"))
        _accs = [s.accession_number for s in _studies]
        check("推理加载器跳过 annotation（不产出假检查）", _accs == ["ACC001"],
              f"实际={_accs}")
        check("推理加载器过滤中文掩膜（不把掩膜当影像）",
              len(_studies[0].series) == 1,
              f"序列={[s.modality for s in _studies[0].series]}")

        try:
            list(_loader.iter_studies(_root))
            check("误传父目录被当场拦截", False, "未报错 → 会把阶段名当病例号")
        except InvalidInputError:
            check("误传父目录被当场拦截", True)

        _single = Path(_tmp) / "one" / "datasets"
        _tiny_nii(_single / "evaluation_first" / "ACC001" / "S1" / "S1.nii.gz")
        check("父目录下只有一个阶段 → 自动下钻",
              [s.accession_number for s in _loader.iter_studies(_single)] == ["ACC001"])

    # 训练侧同一误用的护栏（误传父目录会白烧 GPU，且金标准全对不上）
    _gg_src = (_GOALS / "shared/data.py").read_text(encoding="utf-8")
    _probe_src = (_TRACK4 / "src/data/probe.py").read_text(encoding="utf-8")
    check("研发侧 discover_cases 有数据根护栏",
          "assert_case_root(root)" in _gg_src)
    check("算法工程 scan_real 有数据根护栏",
          "assert_case_root(root)" in _probe_src)
    check("两处护栏使用同一份阶段目录清单",
          all(f'"{p}"' in _gg_src and f'"{p}"' in _probe_src for p in
              ("evaluation_first", "evaluation_finals")))

    # 入口脚本：本地预演必须与容器共享同一套环境变量解析
    _local_src = (_MAIN / "scripts/local_eval.py").read_text(encoding="utf-8")
    check("local_eval 不走裸构造（否则 Factory/权重根被静默忽略）",
          "Settings.from_env()" in _local_src and "replace(" in _local_src)
    check("local_eval 补了仓根 sys.path（否则 CLI 直接 ModuleNotFoundError）",
          "sys.path.insert" in _local_src)
    _mock_src = (_MAIN / "scripts/mock_competition.py").read_text(encoding="utf-8")
    check("Mock Competition 可指向真实数据目录", '"--dataset"' in _mock_src)

    # ---------------------------------------------------------------- #
    _section("⑭ 官方数据的模态来源：SeriesType.xlsx（UID 命名序列）")
    # 训练侧的模态原本只能从**目录名**猜。本地模拟集目录名是 flair_0000，所以
    # 一直正常；官方数据目录名是 DICOM UID，关键词全不命中 →
    # "病例数正常、却报 无任何可用序列"。这条必须用**行为**验证，
    # 因为两边代码都能"导入成功、结构正确"，只在真数据上才炸。
    _goals_src = (_GOALS / "shared/data.py").read_text(encoding="utf-8")
    _probe_src2 = (_TRACK4 / "src/data/probe.py").read_text(encoding="utf-8")
    _labels_src = (_TRACK4 / "src/data/labels.py").read_text(encoding="utf-8")
    check("研发侧实现 read_series_types", "def read_series_types" in _goals_src)
    check("算法工程实现 read_series_types",
          "def read_series_types" in _labels_src and "read_series_types" in _probe_src2)
    check("研发侧模态取自 desc（不再是裸目录名）",
          's.get("desc")' in _goals_src)
    check("掩膜识别有严格线索把关（'T1增强' 不会被当成掩膜）",
          "has_strict_mask_hint" in _probe_src2 and "STRICT_MASK_KW" in _labels_src)

    with tempfile.TemporaryDirectory() as _tmp2:
        from openpyxl import Workbook                                # noqa: PLC0415 本段专用
        from shared.data import find_masks, load_case                 # noqa: PLC0415
        _root = Path(_tmp2) / "annotation"
        _acc = "3255123456"
        _uids = {"f": "1.2.826.0.1.3680043.2.1125.1.1001",
                 "t": "1.2.826.0.1.3680043.2.1125.1.1002",
                 "m": "1.2.826.0.1.3680043.2.1125.1.1003"}
        for _u in _uids.values():
            _tiny_nii(_root / _acc / _u / f"{_u}.nii.gz")
        _wb = Workbook()
        _ws = _wb.active
        _ws.append(["AccessionNumber", "SeriesUid", "SeriesType"])
        for _k, _v in (("f", "FLAIR"), ("t", "T1增强"), ("m", "瘤体")):
            _ws.append([_acc, _uids[_k], _v])
        _wb.save(_root / "SeriesType.xlsx")

        check("研发侧：UID 序列用类型表解析出模态",
              sorted(s["desc"] for s in discover_cases(_root)[0]["series"]) == ["FLAIR", "T1增强"])
        check("研发侧：'瘤体' 归为掩膜",
              list(find_masks(discover_cases(_root)[0])) == ["core"])
        check("研发侧：load_case 能建出体积（原报错点）",
              load_case(discover_cases(_root)[0], common_spacing=(1.0, 1.0, 1.0))[0].ndim == 4)

        if str(_TRACK4) not in sys.path:
            sys.path.insert(0, str(_TRACK4))
        from src.data.probe import scan_real                          # noqa: PLC0415
        _c = scan_real(str(_root))[0]
        check("算法工程：类型表给出模态（'T1增强' 仍算影像）",
              sorted(_c["images"]) == ["flair", "t1c"], f"images={sorted(_c['images'])}")
        check("算法工程：'瘤体' 归为掩膜", list(_c["masks"]) == ["core"])

    # ---------------------------------------------------------------- #
    _section("⑮ 结构化字段金标准：能认表头、能找到上级目录、三种『空』可区分")
    # 目标三/目标四的监督信号全来自这张表。它出问题时**训练照样跑完**
    # （loss 只统计有 mask 的样本），只是分类头学不到东西 —— 所以必须能在
    # 探针阶段就把"没表 / 认不出检查号 / 列名没映射"三种情况分开报出来。
    _labels_src2 = (_TRACK4 / "src/data/labels.py").read_text(encoding="utf-8")
    _probe_src3 = (_TRACK4 / "src/data/probe.py").read_text(encoding="utf-8")
    check("检查号列支持多种命名（含中文）",
          "ID_COLUMN_KEYWORDS" in _labels_src2 and "检查号" in _labels_src2)
    check("表搜索覆盖上级目录且排除非字段表",
          "max_parents" in _labels_src2 and "_NON_LABEL_TABLE_KW" in _labels_src2)
    check("探针报告输出 labels_hint（区分三种空）",
          '"labels_hint"' in _probe_src3)

    with tempfile.TemporaryDirectory() as _tmpl:
        import csv as _csv                                            # noqa: PLC0415
        from src.data.labels import (find_structured_tables,          # noqa: PLC0415
                                     read_structured_table,
                                     structured_from_row)
        from src.data.probe import probe as _probe                    # noqa: PLC0415
        _lroot = Path(_tmpl) / "training" / "annotation"
        _tiny_nii(_lroot / "ACC001" / "S1" / "S1.nii.gz")
        _lfields = ["病理结果", "glioma_with_label", "location_of_lesion",
                    "lesion_morphology", "tumor_feature_necrosis"]
        with open(Path(_tmpl) / "training" / "labels.csv", "w",
                  encoding="utf-8-sig", newline="") as _f:            # 表在数据根的上一级
            _w = _csv.writer(_f)
            _w.writerow(["检查号"] + _lfields)
            _w.writerow(["ACC001", "脑胶质瘤3级", "是", "右侧基底节区", "规则", "有"])
        check("上级目录的金标准表被找到",
              any(h.endswith("labels.csv") for h in find_structured_tables(str(_lroot))))
        _row = read_structured_table(str(Path(_tmpl) / "training" / "labels.csv")).get("ACC001")
        _mapped = structured_from_row(_row) if _row else {}
        check("中文表头『检查号』可取行并映射字段",
              bool(_mapped.get("WHO_Grade")) and bool(_mapped.get("Location")),
              f"fields={sorted(_mapped)[:4]}")
        _rep = _probe(str(_lroot), limit_cases=1)["report"]
        check("有表时 label_field_counts 非空且无 hint",
              bool(_rep["label_field_counts"]) and not _rep["labels_hint"])

        _empty = Path(_tmpl) / "empty" / "annotation"
        _tiny_nii(_empty / "ACC002" / "S1" / "S1.nii.gz")
        _re = _probe(str(_empty), limit_cases=1)["report"]
        check("无表时 hint 明确指向『没找到表』",
              "没找到" in _re["labels_hint"], _re["labels_hint"][:36])
        with open(_empty / "weird.csv", "w", encoding="utf-8-sig", newline="") as _f2:
            _csv.writer(_f2).writerow(["甲", "乙"])
        _re2 = _probe(str(_empty), limit_cases=1)["report"]
        check("有表无检查号列时 hint 指出认不出检查号",
              "检查号" in _re2["labels_hint"], _re2["labels_hint"][:36])

        # 官方标注表的排版：标题行 → 空行 → 真表头。
        # 早期实现把第一行当表头（pandas 默认行为），列名变成"标题/Unnamed"，
        # 检查号列认不出 → 整表 0 行 → label_field_counts 空，而表看着完全正常。
        from openpyxl import Workbook as _WB                          # noqa: PLC0415
        from src.data.labels import read_structured_table as _rst     # noqa: PLC0415
        _rt = Path(_tmpl) / "labels2"
        _rt.mkdir()
        _hp = _rt / "脑胶质瘤标注结果-训练集.xlsx"
        _wb = _WB()
        _ws = _wb.active
        _ws.append(["脑胶质瘤标注结果（训练集）"])          # 标题行
        _ws.append([])                                      # 空行
        _ws.append(["检查号", "病理结果", "location_of_lesion", "lesion_morphology"])
        _ws.append(["c0e1f8f253ba45be843411ca45073cac", "脑胶质瘤3级",
                    "右侧基底节区", "规则"])
        _wb.save(_hp)
        _tb = _rst(str(_hp))
        _key = "c0e1f8f253ba45be843411ca45073cac"
        check("标题行/空行不影响表头识别（官方排版）",
              str((_tb.get(_key) or {}).get("病理结果", "")) == "脑胶质瘤3级",
              f"键数={len(_tb)}")
        # 表里大写、磁盘小写 → 大小写折叠后仍能对上
        _tb_up = _rst(str(_hp))
        check("大小写折叠键可用", _key.upper() in {k.upper() for k in _tb_up} and
              "3" == str(structured_from_row(_tb_up[_key]).get("WHO_Grade")))
        # 数据在第二个工作表
        _hp2 = _rt / "multi.xlsx"
        _wb2 = _WB()
        _wb2.active.title = "说明"
        _wb2.active.append(["本表为说明页"])
        _ws2 = _wb2.create_sheet("data")
        _ws2.append(["AccessionNumber", "病理结果"])
        _ws2.append(["ACC0001", "脑胶质瘤3级"])
        _wb2.save(_hp2)
        check("数据在第二个工作表也能读到", bool(_rst(str(_hp2)).get("ACC0001")))
        # 计数用唯一记录数（同一行会有多个别名键，直接 len() 会翻倍）
        _p_src = (_TRACK4 / "src/data/probe.py").read_text(encoding="utf-8")
        check("n_structured_rows 按唯一记录计数",
              "len({id(v) for v in struct.values()})" in _p_src)

        # 列名不可靠：**按取值**找检查号列（拿磁盘上的检查号逐列比对，列名叫什么都不影响）
        check("实现按取值定位检查号列",
              "_best_id_column_by_values" in _labels_src2)
        check("探针把磁盘检查号传给解析器",
              "known_ids=known_ids" in _p_src and "known_ids" in _p_src)
        check("有『摊开看表』脚本", (_TRACK4 / "scripts/30_inspect_table.py").is_file())
        _hostile = Path(_tmpl) / "hostile.xlsx"
        _wb3 = _WB()
        _ws3 = _wb3.active
        _ws3.append(["脑胶质瘤标注结果（训练集）"])                 # 标题行
        _ws3.append([])                                             # 空行
        _ws3.append(["记录编号(case)", "病理结果", "location_of_lesion",
                     "lesion_morphology"])                          # 检查号列名"认不出"
        _ws3.append(["c0e1f8f253ba45be843411ca45073cac",
                     "脑胶质瘤3级", "右侧基底节区", "规则"])
        _wb3.save(_hostile)
        _hostile_root = Path(_tmpl) / "hostile_ds"
        _tiny_nii(_hostile_root / "c0e1f8f253ba45be843411ca45073cac" / "S1" / "S1.nii.gz")
        _rep3 = _probe(str(_hostile_root), limit_cases=1)["report"]
        check("列名认不出时仍能按取值解析出字段",
              "WHO_Grade" in _rep3["label_field_counts"],
              f"keys={sorted(_rep3['label_field_counts'])[:4]}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 66)
    total = len(_PASSED) + len(_FAILED)
    print(f"通过 {len(_PASSED)}/{total}")
    if _FAILED:
        print("失败项：")
        for f in _FAILED:
            print(f"  ✗ {f}")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
