#!/usr/bin/env python
"""审计插件与训练工程的**结构完整性**：找出"文件在、功能不在"的空壳。

为什么需要这个脚本
------------------
曾经出现过这样的情况：六个 Goal 的 ``dataset.py`` / ``augmentations.py`` /
``losses.py`` / ``evaluate.py`` **只有 docstring 加一个常量**——

    GOAL = "goal1_authenticity"

文件存在、注释写得很完整（"实现已提取到公共库单点维护"），
但 ``from tasks.goal1_authenticity.dataset import build_datasets`` 会直接
**ImportError**。这类空壳不会让任何测试失败，也不会影响推理链路，
因此可以一直潜伏——**"有解释"不等于"有功能"**。

本脚本用两条独立证据来判定：

1. **AST 结构**：文件是否有函数/类定义，或至少是重导出代理；
2. **真实调用**：把每个对外 API 实际调一次，并断言返回值。

用法::

    python scripts/26_audit_plugin_completeness.py
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_TRACK4 = _HERE.parent
_MAIN = pathlib.Path(os.environ.get("GLIOMA_MAIN_ROOT")
                     or (_TRACK4.parent / "Glioma_recognition-main")).resolve()
_GOALS = pathlib.Path(os.environ.get("GLIOMA_GOALS_ROOT")
                      or (_TRACK4.parent / "glioma_goals")).resolve()

GOALS = ["goal1_authenticity", "goal2_stitched", "goal2_duplicate",
         "goal3_tumor", "goal4_diagnosis", "goal5_segmentation"]

#: 包初始化文件允许为空
EMPTY_OK = {"__init__.py"}

_PASSED: list[str] = []
_FAILED: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        _PASSED.append(name)
        print(f"  ✓ {name}")
    except Exception as exc:                                      # noqa: BLE001
        _FAILED.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  ✗ {name}  → {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# 真实调用用的探针（必须在被调用前定义）
# --------------------------------------------------------------------------- #
def _sig_dataset(fn) -> None:
    ps = inspect.signature(fn).parameters
    for want in ("raw", "fold", "limit", "seed", "patch"):
        if want not in ps:
            raise AssertionError(f"build_datasets 缺参数 {want}")


def _call_augment(m) -> None:
    cfg = m.build_augment({"augment": {"flip_prob": 0.42}})
    if cfg.get("flip_prob") != 0.42 or cfg.get("enabled") is not True:
        raise AssertionError(f"build_augment 未透传配置: {cfg}")
    if m.build_augment({}).get("enabled") is not True:
        raise AssertionError("缺配置时未给默认值")


def _call_disabled(m) -> None:
    if m.disabled().get("enabled") is not False:
        raise AssertionError(f"disabled() 未关闭增强: {m.disabled()}")


def _call_cls_spec(m) -> None:
    got = m.cls_spec({"labels": {"fields": [
        {"key": "TumorProbability", "type": "binary"},
        {"key": "WHO_Grade", "type": "multi", "classes": ["1", "2", "3", "4"]},
    ]}})
    if got != [("TumorProbability", 1), ("WHO_Grade", 4)]:
        raise AssertionError(f"cls_spec 结果不对: {got}")


def _call_loss_weights(m) -> None:
    if not isinstance(m.loss_weights(), dict):
        raise AssertionError("loss_weights 未返回 dict")


def _call_loss_fn(m) -> None:
    import torch

    fn = m.build_loss_fn({"loss_weights": {}}, spec=[("TumorProbability", 1)])
    out = {"seg": None, "ds": None, "special": None, "embed": None,
           "cls": [torch.zeros(2, 1, requires_grad=True)]}
    batch = {"labels": torch.tensor([[1.0], [0.0]]), "label_mask": torch.ones(2, 1)}
    loss, _parts = fn(out, batch)
    if not torch.is_tensor(loss) or not torch.isfinite(loss):
        raise AssertionError(f"build_loss_fn 未返回有限张量: {loss}")


def _call_main(m) -> None:
    if not callable(getattr(m, "main", None)):
        raise AssertionError("缺 main()")


def _callable_only(fn) -> None:
    """只断言"导出了可调用入口"（用于两个工程接口有意不同的场景）。"""
    if not callable(fn):
        raise AssertionError("不是可调用对象")


def _sig_dataset_goals(fn) -> None:
    """训练工程侧的签名：``(cfg, data_root, goal_dir, limit, seed)``。

    与提交侧**参数形态不同**（提交侧直接给 ``raw`` + ``fold``），但**口径一致**：
    两侧都优先用同一份 ``folds.json`` 划分验证集。

    ⚠️ 曾经这里写着"训练工程让每个 Goal 自行划分验证集"，那是把"**支持**
    五个人并行做"误读成了"**必须**并行做"。实际要求是：

    * 无论五个 Goal 由五个人并行做、还是一个人串行做，**都要用同一份
      折划分** —— 串行时统一口径才让五个指标可比，并行时统一口径才让
      各 Goal 的权重可以合并/集成；
    * 按 ``val_ratio`` 自行划分只是"还没有 folds.json"时的过渡（会打日志提示）。
    """
    ps = inspect.signature(fn).parameters
    for want in ("cfg", "data_root", "goal_dir", "limit", "seed"):
        if want not in ps:
            raise AssertionError(f"build_datasets 缺参数 {want}")


# --------------------------------------------------------------------------- #
def scan(root: pathlib.Path, label: str) -> list[str]:
    """返回该工程下的空壳文件列表。"""
    print(f"\n{'=' * 74}\n① 结构扫描：{label}（{root}）\n{'=' * 74}")
    shells, n_impl, n_proxy = [], 0, 0
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" in str(f):
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        defs = [n for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        imps = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        if defs:
            n_impl += 1
        elif imps:
            n_proxy += 1
        elif f.name in EMPTY_OK:
            pass
        else:
            shells.append(str(f.relative_to(root)))
    print(f"\n实现 {n_impl} 个 / 重导出代理 {n_proxy} 个 / 空壳 {len(shells)} 个")
    if shells:
        print("✗ 空壳文件（无定义、无重导出）：")
        for s in shells:
            print(f"    {s}")
    else:
        print("✓ 无空壳文件")
    return shells


def _normalized_src(path: pathlib.Path) -> str:
    """去掉 import 与类型注解后的源码转写。

    两个工程的模块路径本就不同（``shared.selector`` vs ``data.series_selector``），
    还会各自加类型注解，因此不能逐字节比较；但**函数体**必须一致——
    这才是"训练出的权重能被推理正确加载"的真正前提。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            node.annotation = None                            # 参数注解：可空
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None                               # 返回注解：可空
        # 注意：不要去改 ``ast.AnnAssign.annotation`` —— 它是**必填字段**，
        # 置 None 会让 ``ast.unparse`` 直接崩（AttributeError: None has no _fields）。
    body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return "\n".join(ast.unparse(n) for n in body)


def check_shared_sources() -> list[str]:
    """两个工程的共享模块必须同源——这是"不合并"能成立的前提。

    它们**不合并**（训练工程不进镜像），因此"训练与推理用的是同一个网络、
    同一套预处理"这件事完全依赖这几份副本的一致性。一旦某侧单独改了
    （比如给骨干加了一层、把 z-score 的裁剪分位改了），训练照样能跑、
    权重照样能存，但推理侧加载出来的结构与训练时不同 —— 而且
    ``load_state_dict(strict=False)`` 只会**静默跳过**不匹配的层。
    """
    print(f"\n{'=' * 74}\n③ 共享模块同源（两工程不合并的前提）\n{'=' * 74}")
    failures: list[str] = []

    #: 必须**逐字节**一致
    strict = [
        ("shared/backbone_mednext.py", "tasks/_common/mednext.py"),
        ("shared/backbone_unet3d.py", "tasks/_common/unet3d.py"),
        ("shared/spatial.py", "tasks/_common/spatial.py"),
        ("shared/sliding.py", "tasks/_common/sliding.py"),
    ]
    for a, b in strict:
        pa, pb = _GOALS / a, _MAIN / b
        if not pa.is_file() or not pb.is_file():
            failures.append(f"{a}: 文件缺失（{pa if not pa.is_file() else pb}）")
            print(f"  ✗ {a} 文件缺失")
            continue
        same = (pa.read_text(encoding="utf-8").rstrip()
                == pb.read_text(encoding="utf-8").rstrip())
        print(f"  {'✓' if same else '✗'} {pathlib.Path(a).name} 逐字节一致")
        if not same:
            failures.append(f"{a} ≠ {b}")

    #: 允许 import / 类型注解差异，但**实现**必须一致
    for a, b in [("shared/volume.py", "tasks/_common/volume.py")]:
        pa, pb = _GOALS / a, _MAIN / b
        same = _normalized_src(pa) == _normalized_src(pb)
        print(f"  {'✓' if same else '✗'} {pathlib.Path(a).name} "
              f"实现一致（允许 import/注解差异）")
        if not same:
            failures.append(f"{a} 实现与 {b} 不一致")

    return failures


def main() -> int:
    shells = scan(_MAIN / "tasks", "提交工程 tasks/")
    shells += scan(_GOALS, "训练工程 glioma_goals/")

    drift = check_shared_sources()
    if drift:
        print("\n✗ 共享模块已漂移（训练与推理可能不在同一套结构/预处理上）：")
        for d in drift:
            print(f"    {d}")

    # ---------------------------------------------------------------- #
    print(f"\n{'=' * 74}\n② 真实调用：每个 Goal 的对外 API（不是『看符号』，而是真调一次）\n{'=' * 74}")
    if str(_MAIN) not in sys.path:
        sys.path.insert(0, str(_MAIN))
    if str(_GOALS) not in sys.path:
        sys.path.insert(0, str(_GOALS))

    for g in GOALS:
        check(f"{g}.dataset.build_datasets 签名",
              lambda g=g: _sig_dataset(
                  importlib.import_module(f"tasks.{g}.dataset").build_datasets))
        check(f"{g}.augmentations.build_augment 真调用",
              lambda g=g: _call_augment(importlib.import_module(f"tasks.{g}.augmentations")))
        check(f"{g}.augmentations.disabled 真调用",
              lambda g=g: _call_disabled(importlib.import_module(f"tasks.{g}.augmentations")))
        check(f"{g}.losses.cls_spec 真调用",
              lambda g=g: _call_cls_spec(importlib.import_module(f"tasks.{g}.losses")))
        check(f"{g}.losses.loss_weights 真调用",
              lambda g=g: _call_loss_weights(importlib.import_module(f"tasks.{g}.losses")))
        check(f"{g}.losses.build_loss_fn 真调用",
              lambda g=g: _call_loss_fn(importlib.import_module(f"tasks.{g}.losses")))
        check(f"{g}.evaluate.main 可调用",
              lambda g=g: _call_main(importlib.import_module(f"tasks.{g}.evaluate")))
        # 训练工程的对应入口（若存在）——签名按各自约定校验，见 _sig_dataset_goals
        if (_GOALS / g / "dataset.py").is_file():
            check(f"{g}.dataset.build_datasets 签名（训练工程约定）",
                  lambda g=g: _sig_dataset_goals(
                      importlib.import_module(f"{g}.dataset").build_datasets))
            # 训练工程的损失接口与提交侧**有意不同**：前者是各 Goal 的
            # 单目标损失（只对本任务那一路回传梯度），后者是统一多任务加权和。
            # 因此这里只断言"是可调用入口"，不套用提交侧的调用方式。
            check(f"{g}.losses.build_loss_fn 可调用（训练工程）",
                  lambda g=g: _callable_only(
                      importlib.import_module(f"{g}.losses").build_loss_fn))

    print(f"\n{'=' * 74}")
    ok = not _FAILED and not shells and not drift
    print(f"通过 {len(_PASSED)}/{len(_PASSED) + len(_FAILED)}"
          f"   空壳 {len(shells)}   同源漂移 {len(drift)}   结论："
          f"{'全绿 ✓' if ok else '存在问题 ✗'}")
    for b in _FAILED:
        print(f"  ✗ {b}")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
