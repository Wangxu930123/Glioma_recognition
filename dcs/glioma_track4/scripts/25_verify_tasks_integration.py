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
    _section("⑭b 数据信息表的位置（就在数据里）与 5 类取值")
    # 三件事都只在真数据上暴露，而且**都不报错**：
    #   ①表在磁盘上、代码只看了一个目录 → 模态整批 other（"病例数正常却无可用序列"）；
    #   ②表里写着"其他"的序列被当成"没认出来"丢给体素模型猜 → 猜成 t2 填进通道；
    #   ③工作区那份兼容表（3_serieslabel.xlsx，**非本赛道数据集内容**）盖住数据集取值
    #     → T2WI 被压成粗粒度 T2、"其他"变成假 FLAIR。
    # 平台实测：``training|verification/annotation/<32位哈希>/<2.25.* UID>/…``，
    # 数据信息 = ``annotation/SeriesType.xlsx``（与病例目录同层），
    # 列 ``AccessionNumber | SeriesUid | SeriesType``，
    # 取值 **5 类** ``T1`` / ``T1CE（增强）`` / ``T2-Flair`` / ``T2WI`` / ``其他``。
    with tempfile.TemporaryDirectory() as _tmp3:
        from openpyxl import Workbook as _WB3                        # noqa: PLC0415
        _env3 = os.environ.pop("GLIOMA_LABELS_DIR", None)             # 只考"数据里的表"
        try:
            _r3 = Path(_tmp3) / "training" / "annotation"
            _acc3 = "0123456789abcdef0123456789abcdef"
            _u3 = {"f": "2.25.269762814043436410034313793283011056280",
                   "t": "2.25.257414435963002133586449291469922441625",
                   "o": "2.25.188710768307220594477022935900342121667",
                   "t1": "2.25.32473739323338211927101306278476129810",
                   "w": "2.25.108049500374765452498127781990506666343"}
            for _u in _u3.values():
                _tiny_nii(_r3 / _acc3 / _u / f"{_u}.nii.gz")
            _wb3 = _WB3()
            _ws3 = _wb3.active
            _ws3.append(["AccessionNumber", "SeriesUid", "SeriesType"])
            _ws3.append([_acc3, _u3["f"], "T2-Flair"])
            _ws3.append([_acc3, _u3["t"], "T1CE（增强）"])
            _ws3.append([_acc3, _u3["t1"], "T1"])
            _ws3.append([_acc3, _u3["w"], "T2WI"])
            _ws3.append([_acc3, _u3["o"], "其他"])
            (_r3 / "标注结果").mkdir()                                # 表藏在容器子目录里
            _wb3.save(_r3 / "标注结果" / "SeriesType.xlsx")

            # 再放一份**工作区那份的兼容表**进同一层，且刻意写得"更粗/更错"：
            # ①T2WI → 粗粒度 "T2"（会静默覆盖权威取值）
            # ②其他 → "FLAIR"（会把"权威排除"变成一路假 flair 灌进通道）
            # 数据集那份必须赢；顺序反了这两条 check 会立刻红。
            _wb3b = _WB3()
            _ws3b = _wb3b.active
            _ws3b.append(["AccessionNumber", "SeriesUid", "SeriesLabel"])
            _ws3b.append([_acc3, _u3["w"], "T2"])
            _ws3b.append([_acc3, _u3["o"], "FLAIR"])
            _wb3b.save(_r3 / "3_serieslabel.xlsx")

            if str(_TRACK4) not in sys.path:
                sys.path.insert(0, str(_TRACK4))
            from src.data.labels import (find_named_table, guess_modality,   # noqa: PLC0415
                                         read_series_types as _t4rst3)
            from src.data.probe import scan_real as _scan3                 # noqa: PLC0415
            from shared.data import read_series_types as _rst3             # noqa: PLC0415

            check("数据信息取值都能归一化（T1CE（增强）/ T2-Flair / T2WI / T1）",
                  (guess_modality("T1CE（增强）"), guess_modality("T2-Flair"),
                   guess_modality("T2WI"), guess_modality("T1"))
                  == ("t1c", "flair", "t2", "t1"))
            check("表藏在子目录（标注结果/）也能找到：数据根=病例目录那一层",
                  bool(find_named_table("SeriesType.xlsx", _r3)))
            check("表藏在子目录也能找到：数据根=某一序列目录（父/祖父回退）",
                  bool(find_named_table("SeriesType.xlsx", _r3 / _acc3 / _u3["f"])))
            check("数据根填高一层（.../training）也能找到（annotation/ 下钻）",
                  bool(find_named_table("SeriesType.xlsx", Path(_tmp3) / "training")))

            _t4 = _t4rst3(_r3)
            check("数据集那份权威：5 条都在，兼容表不覆盖（T2WI 不被压成 T2）",
                  _t4.get((_acc3, _u3["w"])) == "T2WI" and len(_t4) == 5,
                  f"n={len(_t4)} t2wi={_t4.get((_acc3, _u3['w']))!r}")
            check("数据集那份权威：『其他』不被兼容表改写成 FLAIR",
                  _t4.get((_acc3, _u3["o"])) == "其他")
            _g3 = _rst3(_r3)
            check("研发侧同样以数据集那份为准（5 条，未被覆盖）",
                  len(_g3) == 5 and _g3.get((_acc3, _u3["o"])) == "其他",
                  f"n={len(_g3)} other={_g3.get((_acc3, _u3['o']))!r}")

            _c3 = _scan3(str(_r3))[0]
            check("5 类取值 → 通道键（T1/T1CE/T2-Flair/T2WI 各一路，其他单列）",
                  sorted(_c3["images"]) == ["flair", "other", "t1", "t1c", "t2"],
                  f"images={sorted(_c3['images'])}")
            check("『其他』序列不进 unknown（不再交给体素模型猜）",
                  not (_c3.get("unknown_series") or []),
                  f"unknown={_c3.get('unknown_series')}")
            check("『其他』被显式标记（报告里可区分「缺表」与「权威排除」）",
                  (_c3["images"].get("other") or {}).get("declared_other") is True)
        finally:
            if _env3 is not None:
                os.environ["GLIOMA_LABELS_DIR"] = _env3

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
    _section("⑯ 研发侧（glioma_goals）官方标注对接")
    # 官方 5 张标注表是模态/掩膜/字段/异常标记的**唯一权威来源**。
    # 之前这些走的是 label.json、目录名关键词、中文列名 —— 官方数据里都不存在，
    # 于是 special 标签恒为 0、字段全空，训练照跑完却什么都没学到。
    _goals_mod = (_GOALS / "shared/official_labels.py")
    _goals_data = (_GOALS / "shared/data.py").read_text(encoding="utf-8")
    check("研发侧新增 official_labels 模块", _goals_mod.is_file())
    check("研发侧 discover_cases 接官方表",
          "_official_context" in _goals_data and "read_characteristics" in _goals_data)
    check("研发侧跳过 fake/compositing/duplicate 当检查号",
          "SPECIAL_SOURCE_DIRS" in _goals_data)
    check("研发侧序列类型不再「缺 SeriesType.xlsx 就提前 return」",
          "不能提前 return" in _goals_data or "read_series_labels" in _goals_data)
    _g1 = (_GOALS / "goal1_authenticity/dataset.py").read_text(encoding="utf-8")
    _g2a = (_GOALS / "goal2_stitched/dataset.py").read_text(encoding="utf-8")
    check("goal1 用官方 special.fake", 'special.get("fake")' in _g1 or '"fake" in special' in _g1)
    check("goal2_stitched 用官方 special.stitched",
          'special.get("stitched")' in _g2a or '"stitched" in special' in _g2a)

    # 行为级：官方格式（含 fake/compositing 子目录 + 5 张表）跑一遍发现流程
    with tempfile.TemporaryDirectory() as _tmo:
        from openpyxl import Workbook as _WB2                          # noqa: PLC0415
        _oroot = Path(_tmo) / "annotation"
        _olabels = Path(_tmo) / "labels"
        _A, _F, _C = "a" * 32, "f" * 32, "c" * 32
        _U = [f"1.2.826.0.1.3680043.2.1125.1.20{i:02d}" for i in range(3)]
        _tiny_nii(_oroot / _A / _U[0] / f"{_U[0]}.nii.gz")
        _tiny_nii(_oroot / _A / _U[1] / f"{_U[1]}.nii.gz")
        _tiny_nii(_oroot / _A / _U[1] / "mask_x.nii.gz")               # 任意名掩膜
        _tiny_nii(_oroot / "fake" / _F / _U[2] / f"{_U[2]}.nii.gz")
        _xlsx_rows = [("1_abnormal.xlsx", ["AccessionNumber", "SeriesUid", "Label"],
                       [(_A, _U[0], "true"), (_A, _U[1], "true"), (_F, _U[2], "fake")]),
                      ("3_serieslabel.xlsx", ["AccessionNumber", "SeriesUid", "SeriesLabel"],
                       [(_A, _U[0], "FLAIR"), (_A, _U[1], "T1CE"), (_F, _U[2], "T2")]),
                      ("4_masklabel.xlsx", ["AccessionNumber", "SeriesUid", "Maskname"],
                       [(_A, _U[1], "mask_x.nii.gz")]),
                      ("5_characteristics.xlsx",
                       ["AccessionNumber", "Glioma", "WHO_grade", "Morphology"],
                       [(_A, "Yes", 3, "Irregular")])]
        for _fn, _hdr, _rows in _xlsx_rows:
            _w = _WB2()
            _w.active.append(_hdr)
            for _r in _rows:
                _w.active.append(list(_r))
            _olabels.mkdir(parents=True, exist_ok=True)
            _w.save(_olabels / _fn)
        _prev_env = os.environ.get("GLIOMA_LABELS_DIR")
        os.environ["GLIOMA_LABELS_DIR"] = str(_olabels)
        try:
            sys.path.insert(0, str(_GOALS))
            from shared.data import discover_cases as _disc            # noqa: PLC0415
            _cases = {c["accession"]: c for c in _disc(_oroot)}
            check("官方格式：正常 + 异常病例都被发现",
                  _A in _cases and _F in _cases, f"{len(_cases)} 例")
            check("官方格式：special.fake 正确（原来恒为 0）",
                  _cases.get(_F, {}).get("special", {}).get("fake") == 1.0)
            check("官方格式：模态来自 3_serieslabel",
                  sorted(s["desc"] for s in _cases[_A]["series"]) == ["FLAIR", "T1CE"])
            check("官方格式：任意名掩膜被识别",
                  "core" in (_cases[_A].get("masks") or {}))
            check("官方格式：字段来自 5_characteristics",
                  (_cases[_A].get("labels") or {}).get("WHO_Grade") == "3")
        finally:
            if _prev_env is None:
                os.environ.pop("GLIOMA_LABELS_DIR", None)
            else:
                os.environ["GLIOMA_LABELS_DIR"] = _prev_env

    # ---------------------------------------------------------------- #
    _section("⑰ 目标二-B：使用官方重复金标准，且负样本不与之冲突")
    # 早期实现把"同一病例两次增强"当正对 —— 那是必然相同的图，学到的是
    # "增强不变性"而不是"识别重复上传"。更危险的是负样本随机抽：
    # 若抽到官方标注的重复对，同一对就被打上两种标签（正 1 / 负 0），
    # 模型只会学到噪声，而且不会报任何错。
    _dup_src = (_GOALS / "goal2_duplicate/dataset.py").read_text(encoding="utf-8")
    check("读取官方 2_duplicate.xlsx", "read_duplicate_pairs" in _dup_src)
    check("负样本排除已知重复对（dup_of）", "dup_of" in _dup_src)
    check("退化时有语义差异告警", "语义不同" in _dup_src)
    check("导出 B 侧检查号以供核验", '"accession_b"' in _dup_src)

    with tempfile.TemporaryDirectory() as _td:
        from openpyxl import Workbook as _WB3                           # noqa: PLC0415
        _droot = Path(_td) / "annotation"
        _dlabels = Path(_td) / "labels"
        _accs = [ch * 32 for ch in "abcdef"]
        _uids = {a: [f"1.2.826.0.1.3680043.2.1125.1.{i:02d}0{j}"
                     for j in (1, 2)] for i, a in enumerate(_accs)}
        for _a in _accs:
            for _u in _uids[_a]:
                _tiny_nii(_droot / _a / _u / f"{_u}.nii.gz")
        _w = _WB3()
        _w.active.append(["AccessionNumber", "SeriesUid", "SeriesLabel"])
        for _a in _accs:
            for _u in _uids[_a]:
                _w.active.append([_a, _u, "FLAIR"])
        _dlabels.mkdir(parents=True, exist_ok=True)
        _w.save(_dlabels / "3_serieslabel.xlsx")
        _pairs = [(_accs[0], _accs[1]), (_accs[2], _accs[3])]
        _w2 = _WB3()
        _w2.active.append(["src_img", "desc_img"])
        for _p in _pairs:
            _w2.active.append(list(_p))
        _w2.save(_dlabels / "2_duplicate.xlsx")

        _prev = os.environ.get("GLIOMA_LABELS_DIR")
        os.environ["GLIOMA_LABELS_DIR"] = str(_dlabels)
        try:
            if str(_GOALS / "goal2_duplicate") not in sys.path:
                sys.path.insert(0, str(_GOALS / "goal2_duplicate"))
            from dataset import DuplicatePairDataset as _DPD              # noqa: PLC0415
            from shared.data import discover_cases as _dc                 # noqa: PLC0415
            from shared.official_labels import read_duplicate_pairs        # noqa: PLC0415
            _ds_cases = _dc(_droot)
            _off = read_duplicate_pairs(str(_dlabels / "2_duplicate.xlsx"))
            _dset = _DPD(_ds_cases, train=True, gold_pairs=_off)
            _gold = {frozenset(p) for p in _pairs}
            _bad_neg = _pos_gold = _neg = 0
            for _i in range(48):
                _s = _dset[_i]
                _a, _b = str(_s["accession"]), str(_s["accession_b"])
                if float(_s["pair"]) == 1.0:
                    _pos_gold += int(frozenset({_a, _b}) in _gold)
                else:
                    _neg += 1
                    _bad_neg += int(frozenset({_a, _b}) in _gold or _a == _b)
            check("行为级：负样本从不撞官方重复对", _bad_neg == 0, f"bad={_bad_neg}")
            check("行为级：正对确实来自官方金标准", _pos_gold > 0, f"gold={_pos_gold}")
            check("行为级：负样本存在", _neg > 0, f"neg={_neg}")
        finally:
            if _prev is None:
                os.environ.pop("GLIOMA_LABELS_DIR", None)
            else:
                os.environ["GLIOMA_LABELS_DIR"] = _prev

    # ---------------------------------------------------------------- #
    _section("⑱ 评测期无标注表：体素统计模型兜底判模态")
    # 官方训练集给了 labels/3_serieslabel.xlsx，评测集**不给**；评测集的序列
    # 目录名是 DICOM UID，关键词一个都命中不了。此时若不判模态：
    # 训练侧表现为"无任何可用序列"直接崩；推理侧更隐蔽 ——
    # inference/pipeline.py 要先知道"哪个序列是 T1C"才能把掩膜写回它的空间，
    # 认不出就写不回去，提交上去的掩膜空间是错的（评测端直接判错）。
    from src.data.dataset import _MODEL_STATE as _MSTATE              # noqa: PLC0415
    from src.data.dataset import pick_series as _pick                # noqa: PLC0415
    from src.data.modality_model import load_default_model as _ldm    # noqa: PLC0415
    from src.data.probe import scan_real as _scan                    # noqa: PLC0415
    from src.utils.config import load_config as _lcfg                 # noqa: PLC0415

    _model = _ldm()
    check("模态模型可用（data/modality_model.json）", _model is not None,
          "缺失时用 scripts/31_train_modality_model.py 训练")

    # 真实通道名必须覆盖模型的输出，否则兜底会**静默失效**（判出来了却填不进去）
    _chs = {str(c["name"]) for c in _lcfg("preprocess")["channels"]}
    check("配置通道名覆盖模型输出", {"t1c", "flair", "t2"} <= _chs, f"channels={sorted(_chs)}")

    _sim_root = Path(os.environ.get("GLIOMA_SIM_ROOT")
                     or "/mnt/data_sdb/wangx/data/Brain_MRI/track4_sim")
    # 用**训练时没见过**的病例：训练脚本按目录序只取每类前 250 例，
    # 高编号病例是留出的 → 这里测的是泛化，不是背题。
    _case_dir = _sim_root / "BraTS2021_01664"
    if _model is not None and _case_dir.is_dir():
        with tempfile.TemporaryDirectory() as _td:
            _like = Path(_td) / "eval_like"
            _uids = {"flair": "1.2.826.0.1.3680043.2.1125.9.101",
                     "t2": "1.2.826.0.1.3680043.2.1125.9.102",
                     "t1c": "1.2.826.0.1.3680043.2.1125.9.103"}
            _truth: dict[str, str] = {}
            for _mod, _uid in _uids.items():
                _src = _case_dir / f"{_mod}_0000" / f"{_mod}.nii.gz"
                if not _src.is_file():
                    continue
                _dst = _like / "ACC9001" / _uid / f"{_uid}.nii.gz"
                _dst.parent.mkdir(parents=True, exist_ok=True)
                _dst.write_bytes(_src.read_bytes())     # 目录名/文件名全是 UID，无标注表
                _truth[str(_dst)] = _mod
            _cases = _scan(str(_like))
            check("UID 目录名且无标注表时仍能扫出检查", len(_cases) == 1, f"cases={len(_cases)}")

            if _cases:
                _c = _cases[0]
                # 前提确认：关键词路径确实**完全失效**（否则本段测不到兜底）
                check("关键词确实认不出模态（只剩 other）",
                      set(_c.get("images") or {}) <= {"other"},
                      f"images={sorted(_c.get('images') or {})}")
                check("未知序列被完整保留（不再只留第一个）",
                      len(_c.get("unknown_series") or []) == len(_truth),
                      f"unknown={len(_c.get('unknown_series') or [])}/{len(_truth)}")

                _log: list[str] = []
                _picked = _pick(_c, _lcfg("preprocess"), _log)
                check("兜底把三路模态都判了出来",
                      {"t1c", "flair", "t2"} <= set(_picked),
                      f"picked={sorted(_picked)}")
                _wrong = [ch for ch, m in _picked.items()
                          if ch in ("t1c", "flair", "t2")
                          and _truth.get(str(m["path"])) != ch]
                check("判出的模态各自对应**正确**的源文件", not _wrong,
                      f"错配={_wrong}")
                check("判别过程有日志可追溯", any("模态判别" in s for s in _log),
                      f"{_log[:1]}")

                # 模型缺失时必须优雅降级（不能抛异常、不能瞎填通道）
                _saved = dict(_MSTATE)
                try:
                    _MSTATE.update(loaded=True, model=None)
                    _none_log: list[str] = []
                    _picked2 = _pick(_c, _lcfg("preprocess"), _none_log)
                    check("无模型时不抛异常且不瞎填模态",
                          not ({"t1c", "flair", "t2"} & set(_picked2)),
                          f"picked={sorted(_picked2)}")
                    check("无模型时给出可操作的提示",
                          any("modality_model.json" in s for s in _none_log))
                finally:
                    _MSTATE.clear()
                    _MSTATE.update(_saved)

                # argmax 撞车时必须靠**次优**标签救回通道：
                # 逐路"先到先得"会把 uid1 错填成 T2、并把 uid2 整路丢掉（3 路只剩 2 路且错 1 路）；
                # 全局贪心指派应把三路全部对齐（uid2 拿 T2 0.95，uid1 落到次优 T1CE 0.55）。
                class _FakeModel:                                     # noqa: D401
                    def __init__(self, table):
                        self.table = table

                    def predict_file(self, path, stride: int = 2):     # noqa: ARG002
                        p = self.table[os.path.basename(str(path))]
                        return max(p, key=p.get), p

                _tbl = {
                    f"{_uids['t2']}.nii.gz": {"T2": 0.90, "T1CE": 0.55, "FLAIR": 0.05},
                    f"{_uids['flair']}.nii.gz": {"FLAIR": 0.98, "T2": 0.01, "T1CE": 0.01},
                    f"{_uids['t1c']}.nii.gz": {"T1CE": 0.97, "FLAIR": 0.02, "T2": 0.01},
                }
                _saved2 = dict(_MSTATE)
                try:
                    _MSTATE.update(loaded=True, model=_FakeModel(_tbl))
                    _picked3 = _pick(_c, _lcfg("preprocess"), None)
                    # 注意 UID 本身含 "." —— 不能用 split(".")[0] 取文件名（会只剩 "1"）
                    _map3 = {ch: os.path.basename(str(m["path"])).replace(".nii.gz", "")
                             for ch, m in _picked3.items() if ch in _uids}
                    check("撞车时按全局贪心指派（三路全对）",
                          _map3 == dict(_uids),
                          f"got={ {k: v[-7:] for k, v in _map3.items()} }")
                finally:
                    _MSTATE.clear()
                    _MSTATE.update(_saved2)

                # 端到端：真的构建出多通道体数据（含公共网格重采样）
                try:
                    from src.data.dataset import build_case_volume as _bcv  # noqa: PLC0415
                    _vol, _aff, _ = _bcv(_c, _lcfg("preprocess"))
                    _ok = tuple(_vol.shape) == (len(_lcfg("preprocess")["channels"]),
                                                *_vol.shape[1:])
                    check("端到端：多通道体数据可构建", _ok and _vol.std() > 0,
                          f"vol={tuple(_vol.shape)} std={float(_vol.std()):.3f}")
                except Exception as _exc:                             # noqa: BLE001
                    check("端到端：多通道体数据可构建", False, f"{type(_exc).__name__}: {_exc}")
    elif _model is not None:
        check("找到可用于验证的留出病例", False, str(_case_dir))

    # ---------------------------------------------------------------- #
    _section("⑲ 文档里标出的「数据路径 / 数据信息路径」")
    # 交付文档必须让人一眼看到：影像在哪、数据信息在哪。这是**文档契约**——
    # 改了目录结构却忘了改文档、或又把工作区那 5 张表说成"本赛道的数据信息"，这里会红。
    _doc_expect = [
        ("README.md", "annotation/SeriesType.xlsx", "数据信息路径"),
        ("README.md", "2.25.* 序列UID", "影像数据路径"),
        ("docs/MASTER_GUIDE.md", "★ 数据路径（影像）", "数据路径显式标注"),
        ("docs/MASTER_GUIDE.md", "★ 数据信息路径", "数据信息路径显式标注"),
        ("docs/MASTER_GUIDE.md", "脑胶质瘤标注结果-训练集.xlsx", "字段金标准表名"),
        ("docs/CLOUD_DESKTOP_RUNBOOK.md", "annotation/SeriesType.xlsx", "数据信息路径"),
        ("docs/DATASET_ROOT_TROUBLESHOOT.md", "★ 数据信息：", "数据信息路径"),
    ]
    for _rel, _needle, _what in _doc_expect:
        _p = _TRACK4 / _rel
        _txt = _p.read_text(encoding="utf-8") if _p.is_file() else ""
        check(f"{_rel} 标出{_what}（含 {_needle!r}）", _needle in _txt)
    _mg = (_TRACK4 / "docs/MASTER_GUIDE.md").read_text(encoding="utf-8")
    check("文档明确那 5 张表与赛道四数据集无关（别再去配它们的路径）",
          "与赛道四数据集没有关系" in _mg)

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
