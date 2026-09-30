#!/usr/bin/env python3
"""提交前**一键预检**：跑完所有关卡，最后给一个 ``GO`` / ``NO-GO``。

设计目标：**一条命令、无需参数、退出码即结论**。
任何一关抛异常都不会让脚本崩 —— 只会把那一关记成 FAIL 并继续跑下一关，
这样你能一次看到**全部**问题，而不是修一个跑一次。

关卡::

  G1 环境     torch / CUDA / 权重文件 / 数据目录 / 官方序列表
  G2 权重     ``load_state_dict`` 后 **``seg`` 头是否真的加载上**（最容易静默出错的一关）
  G3 预处理   Goal5Config 与训练侧 preprocess.yaml 的 16 项参数 + 通道取用链
  G4 冒烟     直接对前 N 例推理，统计**空掩膜率**与 ``pmax``
  G5 端到端   完整 runner（Writer+Validator）跑一批，再离线校验输出格式（``--full`` 才跑）

用法::

  python3 scripts/pre_submit_check.py                    # 默认 5 例冒烟，~1 分钟
  python3 scripts/pre_submit_check.py --limit 20         # 多冒烟几例更可靠
  python3 scripts/pre_submit_check.py --full             # 完整跑（慢，提交前最后一遍）

退出码：``0`` = GO，``1`` = NO-GO。
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DEFAULT = "/2026aicompetition/datasets/verification/original"
TRAIN_CFG_DEFAULT = "/2026aicompetition/workspace/dcs/glioma_track4/configs/preprocess.yaml"

#: 关卡结果：(名称, 状态, 摘要)。状态 ∈ {PASS, FAIL, WARN, SKIP}
_GATES: list[tuple[str, str, str]] = []

#: 阻止 GO 的关卡名。``WARN`` 与 ``SKIP`` 不阻止。
_BLOCKING: set[str] = set()


def _record(name: str, status: str, summary: str = "") -> None:
    _GATES.append((name, status, summary))
    if status == "FAIL":
        _BLOCKING.add(name)
    icon = {"PASS": "✔", "FAIL": "✘", "WARN": "!", "SKIP": "-"}.get(status, "?")
    print(f"[{icon}] {name:<18} {summary}", flush=True)


def _guard(name: str):
    """把一关包成 try/except：内部抛错 → 记 FAIL，不中断其余关卡。"""
    def deco(fn):
        def wrapper(*a, **kw):
            print()
            print("-" * 88)
            print(f"▶ {name}")
            print("-" * 88)
            try:
                return fn(*a, **kw)
            except Exception as exc:                              # noqa: BLE001
                _record(name, "FAIL", f"关卡自身抛错 {type(exc).__name__}: {exc}")
                traceback.print_exc()
                return None
        return wrapper
    return deco


# --------------------------------------------------------------------------- #
# G1 环境
# --------------------------------------------------------------------------- #
@_guard("G1 环境")
def gate_env(dataset: Path, limit: int) -> dict:
    info: dict = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["device"] = torch.cuda.get_device_name(0) if info["cuda"] else "cpu"
    except Exception as exc:                                      # noqa: BLE001
        _record("G1 环境", "FAIL", f"import torch 失败: {exc}")
        return info

    from core.config import Settings
    from tasks.goal5_segmentation.config import Goal5Config
    from tasks.goal5_segmentation.inference import resolve_ckpts
    from tasks.goal5_segmentation.preprocess import CHANNEL_FALLBACK, CHANNEL_ORDER

    settings = Settings.from_env()
    cfg = Goal5Config()
    info["settings"] = settings
    info["cfg"] = cfg

    print(f"torch      : {info['torch']}")
    print(f"CUDA       : {info['cuda']}  {info['device']}")
    print(f"权重根     : {settings.ckpt_root}")
    print(f"数据根     : {dataset}  存在={dataset.is_dir()}")
    print(f"取例上限   : {limit}")

    if not info["cuda"]:
        _record("G1 环境", "WARN", "CUDA 不可用 → 会用 CPU 推理（极慢，正式评测必须用 GPU）")
    if not dataset.is_dir():
        _record("G1 环境", "FAIL", f"数据目录不存在: {dataset}")
        return info

    try:
        paths = resolve_ckpts(settings.ckpt_root, cfg.core_ckpt_rel)
        info["ckpt_paths"] = paths
        size_mb = sum(p.stat().st_size for p in paths) / 1e6
        print(f"权重文件   : {len(paths)} 个，共 {size_mb:.1f} MB")
        for p in paths:
            print(f"             {p}")
    except Exception as exc:                                      # noqa: BLE001
        _record("G1 环境", "FAIL", f"权重解析失败: {type(exc).__name__}: {exc}")
        return info

    print(f"通道取用链 : " + ", ".join(
        f"{n}<-{list(CHANNEL_FALLBACK[n])}" for n in CHANNEL_ORDER))

    # ---- 插件工厂：不设 = 服务**静默跑 Dummy 基线**（本仓最致命的一条）----
    # ``core/registry.build_pipeline(None)`` 只打印一条告警就返回空 Pipeline：
    # 服务照常起来、/health 通、回调正常，但答案是占位内容。评测不可重跑 → 直接 0 分。
    factory = os.environ.get("COMPETITION_PIPELINE_FACTORY") or settings.pipeline_factory
    if not factory:
        _record("G1 环境", "FAIL",
                "COMPETITION_PIPELINE_FACTORY 未设置 → 服务会**静默跑 Dummy 基线**"
                "（答案全是占位内容）。用 ./start.sh 启动（已默认设为 "
                "tasks.real_pipeline:build_pipeline），或手动 export")
        return info
    print(f"插件工厂   : {factory}")
    if "real_pipeline" not in factory and "glioma.pipeline" not in factory:
        print("             ⚠️ 不是已知的两个真实工厂，确认它真的注册了 Goal5")

    if "G1 环境" not in _BLOCKING:
        _record("G1 环境", "PASS",
                f"torch {info['torch']} / {'cuda' if info['cuda'] else 'cpu'} / "
                f"{len(paths)} 份权重 / 数据就位")
    return info


# --------------------------------------------------------------------------- #
# G2 权重：seg 头是否真的加载上
# --------------------------------------------------------------------------- #
@_guard("G2 权重")
def gate_weights(info: dict) -> None:
    import torch
    from tasks.goal5_segmentation.models.factory import build_goal5_models

    paths = info.get("ckpt_paths") or []
    cfg = info.get("cfg")
    if not paths or cfg is None:
        _record("G2 权重", "FAIL", "G1 未取到权重路径 → 跳过")
        return

    problems: list[str] = []
    for p in paths:
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        if not isinstance(ck, dict):
            problems.append(f"{p.name}: 顶层不是 dict（是 {type(ck).__name__}）")
            continue
        meta = [k for k in ("arch", "model_cfg", "thresholds", "epoch",
                            "best_metric", "fold") if k in ck]
        print(f"\n{p.name}  大小={p.stat().st_size / 1e6:.1f}MB")
        print(f"  元数据键 : {meta}")
        for k in ("arch", "thresholds", "global_size", "epoch", "best_metric", "fold"):
            if k in ck:
                print(f"    {k} = {ck[k]}")
        if "model_cfg" in ck:
            print(f"    model_cfg = {ck['model_cfg']}")

        state = ck.get("model_ema") or ck.get("model") or ck
        n_tensor = sum(1 for v in state.values() if hasattr(v, "shape"))
        print(f"  张量数   : {n_tensor}")
        if not n_tensor:
            problems.append(f"{p.name}: state_dict 里没有张量 → 不是模型权重")
            continue

        m = build_goal5_models(cfg, ck.get("cls_spec") or [], shared=True)
        missing, unexpected = m.core_model.load_state_dict(state, strict=False)
        buckets: dict[str, int] = {}
        for k in missing:
            buckets[k.split(".")[0]] = buckets.get(k.split(".")[0], 0) + 1
        print(f"  缺失键   : {len(missing)} 个 {buckets or '（无）'}")
        print(f"  多余键   : {len(unexpected)} 个")

        seg_missing = [k for k in missing if "seg" in k.lower()]
        backbone_missing = [k for k in missing
                            if k.startswith(("enc", "dec", "bottleneck", "stem"))]
        if seg_missing:
            print("  !!! **seg 头没加载上 → 分割头是随机初始化的**")
            print(f"  !!! 样例: {seg_missing[:5]}")
            problems.append(f"{p.name}: seg 头缺失 {len(seg_missing)} 个键（随机初始化）")
        if backbone_missing:
            problems.append(f"{p.name}: 骨干缺失 {len(backbone_missing)} 个键")

        for key in sorted(state):
            if "seg" in key.lower() and key.endswith("weight") and hasattr(state[key], "float"):
                f = state[key].detach().float()
                if f.dim() >= 1:
                    print(f"  {key}: mean={f.mean():+.4f} std={f.std():.4f}")
        del ck

    if problems:
        _record("G2 权重", "FAIL", "；".join(problems))
    else:
        _record("G2 权重", "PASS", "seg 头与骨干全部加载成功，权重是训练产物")


# --------------------------------------------------------------------------- #
# G3 预处理一致性
# --------------------------------------------------------------------------- #
@_guard("G3 预处理")
def gate_preprocess(train_cfg: Path, info: dict) -> None:
    from tasks.goal5_segmentation.config import Goal5Config
    from tasks.goal5_segmentation.preprocess import CHANNEL_FALLBACK, CHANNEL_ORDER

    cfg = info.get("cfg") or Goal5Config()
    if not train_cfg.is_file():
        _record("G3 预处理", "WARN",
                f"训练配置不存在({train_cfg}) → **无法判定**，只能打印提交侧取值")
        print(f"提交侧: max_spacing_factor={cfg.max_spacing_factor} overlap={cfg.overlap} "
              f"patch={cfg.patch} min_tumor_voxels={cfg.min_tumor_voxels} "
              f"keep_components={cfg.keep_components} bridge_mm={cfg.bridge_mm}")
        return

    try:
        import yaml
        train = yaml.safe_load(train_cfg.read_text(encoding="utf-8")) or {}
    except Exception as exc:                                      # noqa: BLE001
        _record("G3 预处理", "WARN", f"YAML 解析失败({exc}) → 无法判定")
        return

    def dig(d, dotted):
        node = d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def norm(v):
        if isinstance(v, (list, tuple)):
            return tuple(norm(x) for x in v)
        if isinstance(v, float):
            return round(v, 6)
        return v

    sub = {
        "geometry.common_spacing": cfg.common_spacing,
        "geometry.max_spacing_factor": cfg.max_spacing_factor,
        "geometry.resample_order_img": 1,
        "geometry.brain_margin_vox": 4,
        "intensity.clip_percentile": (0.5, 99.5),
        "intensity.foreground_only": True,
        "inference.patch": cfg.patch,
        "inference.overlap": cfg.overlap,
        "inference.tta_flips": cfg.tta_flips,
        "inference.seg_tta_flips": cfg.tta_flips,
        "inference.tta_batch": cfg.tta_batch,
        "inference.global_size": cfg.global_size,
        "inference.min_tumor_voxels": cfg.min_tumor_voxels,
        "inference.keep_components": cfg.keep_components,
        "inference.bridge_mm": cfg.bridge_mm,
    }
    bad: list[str] = []
    print(f"{'参数':<36}{'训练侧':>18}{'提交侧':>18}   判定")
    for k, s in sub.items():
        t = norm(dig(train, k))
        s = norm(s)
        mark = "OK" if t == s else ("训练侧未定义" if t is None else "!! 不一致")
        if mark == "!! 不一致":
            bad.append(k)
        print(f"{k:<36}{str(t):>18}{str(s):>18}   {mark}")

    chans = train.get("channels") or []
    want = {str(c.get("name")): tuple([c.get("name")] + list(c.get("fallback") or []))
            for c in chans if isinstance(c, dict)}
    for name in CHANNEL_ORDER:
        got = tuple(CHANNEL_FALLBACK[name])
        exp = want.get(name)
        if exp is not None and exp != got:
            bad.append(f"channels.{name}")
            print(f"channels.{name:<26}{str(exp):>18}{str(got):>18}   !! 不一致")
        else:
            print(f"channels.{name:<26}{str(exp):>18}{str(got):>18}   OK")

    if bad:
        _record("G3 预处理", "FAIL", f"{len(bad)} 处与训练不一致: {bad}")
    else:
        _record("G3 预处理", "PASS", f"{len(sub)} 项参数 + 通道链全部与训练侧一致")


# --------------------------------------------------------------------------- #
# G4 冒烟：空掩膜率
# --------------------------------------------------------------------------- #
@_guard("G4 冒烟")
def gate_smoke(dataset: Path, limit: int, info: dict) -> None:
    from data.loader import DatasetLoader
    from tasks.goal5_segmentation.task import Goal5Task

    class _Ctx:
        def __init__(self, study):
            self.study = study
            self.warnings: list[str] = []
            self.diagnostics: dict = {}

    task = Goal5Task(settings=info["settings"])
    task.load_model()
    print(f"阈值 : {[round(float(t), 3) for t in task._loaded.thresholds]}")
    print(f"滑窗 : patch={task.cfg.patch} overlap={task.cfg.overlap} "
          f"bridge={task.cfg.bridge_mm}min={task.cfg.min_tumor_voxels}")
    print("=" * 88)

    seen = empty = 0
    pmax_core: list[float] = []
    pmax_flair: list[float] = []
    pp_killed = 0
    failures: list[str] = []
    for study in DatasetLoader().iter_studies(dataset):
        ctx = _Ctx(study)
        try:
            task.predict(ctx)
        except Exception as exc:                                  # noqa: BLE001
            failures.append(f"{study.accession_number}: {type(exc).__name__}: {exc}")
            print(f"[{study.accession_number}] ✗ {type(exc).__name__}: {exc}")
            continue
        d = ctx.diagnostics.get("goal5") or {}
        seen += 1
        mp = d.get("max_probs") or [0.0, 0.0]
        th = d.get("thresholds") or [0.5, 0.5]
        pmax_core.append(float(mp[0]))
        pmax_flair.append(float(mp[1]))
        is_empty = not d.get("core_voxels") and not d.get("flair_voxels")
        empty += is_empty
        pre = (d.get("core_pre_voxels") or 0) + (d.get("flair_pre_voxels") or 0)
        pp_killed += bool(is_empty and pre > 0)
        print(f"[{study.accession_number}] missing={d.get('missing_channels')} "
              f"pmax={mp} thr={th} core={d.get('core_voxels')}"
              f"(pre {d.get('core_pre_voxels')}) flair={d.get('flair_voxels')}"
              f"(pre {d.get('flair_pre_voxels')})"
              f"{'  ← 空' if is_empty else ''}")
        for w in ctx.warnings:
            print(f"    warn: {w}")
        if seen >= limit:
            break

    import statistics
    print("=" * 88)
    print(f"推理成功 : {seen} 例 | 异常 {len(failures)} 例")
    if seen:
        rate = empty / seen
        print(f"空掩膜   : {empty}/{seen} = {rate:.0%}")
        print(f"其中被后处理吃掉 : {pp_killed} 例")
        print(f"pmax core  : 均值 {statistics.fmean(pmax_core):.3f} "
              f"最大 {max(pmax_core):.3f}")
        print(f"pmax flair : 均值 {statistics.fmean(pmax_flair):.3f} "
              f"最大 {max(pmax_flair):.3f}")
    else:
        rate = 1.0
        print("!! 一例都没跑成功")

    if failures:
        _record("G4 冒烟", "FAIL",
                f"{len(failures)} 例抛异常（首条: {failures[0][:80]}）")
    elif seen == 0:
        _record("G4 冒烟", "FAIL", "0 例成功")
    elif rate > 0.6:
        _record("G4 冒烟", "FAIL",
                f"空掩膜率 {rate:.0%}（>60%）→ 提交上去几乎必然低分，先别交")
    elif rate > 0.2:
        _record("G4 冒烟", "WARN",
                f"空掩膜率 {rate:.0%}（20%~60%）→ 可提交但分数受损，建议先看诊断")
    else:
        _record("G4 冒烟", "PASS", f"空掩膜率 {rate:.0%}（{empty}/{seen}）")


# --------------------------------------------------------------------------- #
# G5 端到端（完整 runner + 离线校验）
# --------------------------------------------------------------------------- #
@_guard("G5 端到端")
def gate_end_to_end(dataset: Path, info: dict, out_root: Path) -> None:
    import uuid
    from dataclasses import replace

    from core.runner import EvaluationJob, EvaluationRunner
    from scripts.validate_output import validate

    settings = replace(
        info["settings"],
        workspace=out_root.parent,
        answer_root=out_root,
        log_root=out_root.parent / "logs",
        callback_url=None,
    )
    runner = EvaluationRunner(settings)
    evaluation_id = "pre-submit"
    result = runner.run(
        EvaluationJob(
            request_id=f"pre-submit-{uuid.uuid4()}",
            evaluation_id=evaluation_id,
            dataset_path=dataset,
        ),
        send_callback=False,
    )
    print(f"产出目录 : {result}")
    rc = validate(Path(result), None)
    if rc != 0:
        _record("G5 端到端", "FAIL", f"输出格式校验未通过（{result}）")
    else:
        _record("G5 端到端", "PASS", f"完整链路 + 输出格式校验通过（{result}）")


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="提交前一键预检（GO / NO-GO）")
    ap.add_argument("--dataset", default=DATA_DEFAULT, help=f"数据根（默认 {DATA_DEFAULT}）")
    ap.add_argument("--limit", type=int, default=5, help="冒烟例数（默认 5）")
    ap.add_argument("--train-config", default=TRAIN_CFG_DEFAULT)
    ap.add_argument("--full", action="store_true",
                    help="额外跑完整 runner 一遭（慢，但这是提交前的最后一道保险）")
    ap.add_argument("--out", default="/tmp/pre_submit_answer", help="--full 时的产出目录")
    a = ap.parse_args(argv)

    dataset = Path(a.dataset).expanduser()
    print("=" * 88)
    print("提交前预检 —— 一条命令给出 GO / NO-GO")
    print("=" * 88)

    info = gate_env(dataset, a.limit) or {}
    if info:
        gate_weights(info)
        gate_preprocess(Path(a.train_config).expanduser(), info)
        if dataset.is_dir():
            gate_smoke(dataset, a.limit, info)
        if a.full and dataset.is_dir():
            gate_end_to_end(dataset, info, Path(a.out).expanduser())

    print()
    print("=" * 88)
    print("汇总")
    print("=" * 88)
    for name, status, summary in _GATES:
        icon = {"PASS": "✔", "FAIL": "✘", "WARN": "!", "SKIP": "-"}.get(status, "?")
        print(f"  [{icon}] {name:<18} {summary}")

    go = not _BLOCKING
    print()
    print("=" * 88)
    if go:
        warn = [n for n, s, _ in _GATES if s == "WARN"]
        print(f"结论：**GO** —— 可以提交" + (f"（有 {len(warn)} 项告警：{warn}）" if warn else ""))
    else:
        print(f"结论：**NO-GO** —— 以下关卡未通过，先别提交：{sorted(_BLOCKING)}")
        print("     每关的详细输出在上面对应小节里；把整份输出贴回来即可定位。")
    print("=" * 88)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
