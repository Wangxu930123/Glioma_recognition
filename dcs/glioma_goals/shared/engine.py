"""共享训练引擎：通用训练循环（EMA / AMP / 梯度裁剪 / warmup 余弦）。

**各 Goal 共用本引擎，但损失由各 Goal 自己注入**：
``train(..., loss_fn)`` 里的 ``loss_fn(out, batch) -> (loss_tensor, parts_dict)``
来自调用方的 ``losses.py``。这样六个任务可以各写各的损失，
而训练循环（显存管理、EMA、恢复、日志）只有一份实现——
训练循环是最容易"各自微调后结果不可比"的地方。

引擎自带三项工程保护：

1. **EMA**：验证与提交都用指数滑动平均权重，小数据下比裸权重稳定；
2. **AMP + 梯度裁剪**：3D 分割显存紧张，且早期梯度易爆；
3. **配对前向**：本任务族里有"重复影像"这种配对任务，需要两次前向；
   引擎在检测到 ``image_b`` 时自动补一次并挂上 ``embed_a``/``embed_b``，
   避免各 Goal 各写一套训练循环。
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from shared.losses import LossWeights


@dataclass
class TrainConfig:
    """训练超参（各 Goal 用自己的 config.yaml 覆盖）。"""

    epochs: int = 100
    batch_size: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp_dtype: str = "bf16"
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    warmup_epochs: int = 3
    num_workers: int = 4
    patch: tuple[int, int, int] = (96, 96, 96)
    seed: int = 42
    out_dir: str = "checkpoints"
    tag: str = "run"
    save_every: int = 10
    #: 每轮验证的批数上限（验证太慢会让训练时间翻倍）
    val_batches: int = 20


def _lr_at(epoch: int, cfg: TrainConfig) -> float:
    """warmup + 余弦退火（小 batch 下早期大学习率会打乱 BN 统计量）。"""
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / max(1, cfg.warmup_epochs)
    t = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * min(1.0, t)))


def _selection_score(val: dict, metrics: dict) -> float:
    """挑选 best 权重的依据（**必须返回有限值**）。

    为什么不能写成 ``float(val.get("score", ...) or 0.0)``：
    ``nan`` 在 Python 里是 **truthy**，``nan or 0.0`` 得到的仍是 ``nan``；
    于是 ``nan > best_score`` 永远为 False，``_save`` **一次都不会执行** ——
    训练全程跑完却没有任何权重文件，而日志照常打印"完成"、
    ``history.jsonl`` 也记成成功，属于最难察觉的失效。

    而"验证指标是 nan"在这里**不是罕见情况**：目标一/二/三的正样本极稀疏
    （本地 fake 6 例、Composition 6 例、金标准对 12 对），验证折里常常
    一个正样本都没有，AUC 必然为 nan。

    兜底顺序：有效的 ``score`` → 训练损失取负（损失越小越好）→ 0.0。
    """
    s = val.get("score")
    if s is not None:
        s = float(s)
        if math.isfinite(s):
            return s
    total = metrics.get("total")
    if total is not None and math.isfinite(float(total)):
        return -float(total)
    return 0.0


class EMA:
    """指数滑动平均权重。"""

    def __init__(self, model, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def update(self, model) -> None:
        import torch

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                        alpha=1 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return self.shadow


def train(model, train_loader, val_fn: Callable | None, cfg: TrainConfig,
          cfg_raw: dict, loss_fn: Callable, device: str = "cuda") -> dict[str, Any]:
    """执行训练并返回 ``{"best", "history", "ckpt"}``。

    Args:
        loss_fn: 各 Goal 的 ``losses.build_loss_fn(cfg_raw)`` 返回值，
                 签名 ``(out, batch) -> (loss_tensor, parts_dict)``。
        cfg_raw: 原始配置字典（``loss_fn`` 内部可能读取 ``loss_weights``）。
    """
    import torch

    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.to(dev).train()
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp is torch.float16))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 初值取 -inf 而非 -1.0：保证**第一个 epoch 结束一定保存一次权重**。
    # 否则若选择指标恒为 nan（见 ``_selection_score``），可能整轮训练
    # 一个权重文件都不产出，而"没有任何产物"很难从日志上看出来。
    best_score, history = -float("inf"), []

    for epoch in range(cfg.epochs):
        lr = _lr_at(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        t0, agg, n_batch = time.time(), {}, 0

        for step, batch in enumerate(train_loader):
            batch = {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp, enabled=(dev == "cuda")):
                out: dict = {}

                # ① 分割头（seg / ds）走 patch：它需要局部高分辨率。
                #    ⚠️ **仅当本任务提供分割监督时**才前向。
                #    配对任务（目标二-B）是纯全局任务、只用 embed，但它的数据集
                #    同样继承自 ``BaseCaseDataset``；若无条件前向一个 96³ patch，
                #    这次前向的激活会与后面的两次整脑前向**同时驻留**，
                #    实测在 24GB 卡上直接 OOM（20.17 GiB）——而实际只需要
                #    两次整脑前向。判据用 ``target``：分割监督必然带它。
                if batch.get("target") is not None:
                    out = model(batch["image"])

                # ② 全局头（cls / special / embed）走**整脑视图**。
                #    这三路的监督信号都是**检查级**的（有无肿瘤、是否拼接、
                #    是否假人体、是否重复），用 96mm 局部 patch 去判断整个检查
                #    会丢失关键证据；而推理侧固定用 192mm 整脑视图
                #    （``tasks/_common/volume.global_view``），训练若用 patch
                #    则输入分布不一致 —— 训练指标好看、上线后接近随机。
                if batch.get("image_global") is not None:
                    out_g = model(batch["image_global"])
                    for _k in ("cls", "special", "embed"):
                        if out_g.get(_k) is not None:
                            out[_k] = out_g[_k]
                    del out_g

                # ③ 配对任务：第二条同样走整脑视图（嵌入必须与推理同尺度）
                if "image_b" in batch:
                    second = batch.get("image_global_b")
                    out_b = model(second if second is not None else batch["image_b"])
                    out["embed_a"] = out.get("embed")
                    out["embed_b"] = out_b.get("embed")
                    del out_b
                loss, parts = loss_fn(out, batch)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            ema.update(model)

            for k, v in (parts or {}).items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n_batch += 1
            if (step + 1) % 20 == 0:
                avg = {k: round(v / n_batch, 4) for k, v in agg.items()}
                print(f"[{cfg.tag}] ep{epoch} step{step + 1}/{len(train_loader)} {avg}",
                      flush=True)

        metrics = {k: v / max(1, n_batch) for k, v in agg.items()}
        ema_model = _with_ema(model, ema)
        val = val_fn(ema_model) if val_fn else {}
        val = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (val or {}).items()}
        score = _selection_score(val, metrics)
        metrics.update({f"val_{k}": v for k, v in val.items() if isinstance(v, (int, float))})
        metrics["lr"], metrics["epoch"] = lr, epoch
        history.append(metrics)

        summary = " ".join(f"{k}={v:.4f}" for k, v in val.items()
                           if isinstance(v, (int, float)))
        # 训练侧展示各损失分项之和（parts 的键由各 Goal 定义，没有统一的 "total"）
        loss_txt = " ".join(f"{k}={metrics[k]:.4f}" for k in metrics
                            if k.startswith(("seg", "cls", "special", "embed", "ds"))
                            and isinstance(metrics[k], (int, float)))
        print(f"[{cfg.tag}] epoch {epoch} {loss_txt} | {summary} "
              f"({time.time() - t0:.0f}s)", flush=True)

        if score > best_score:
            best_score = score
            _save(out_dir / "best.pth", model, ema, cfg, cfg_raw, best_score, epoch)
        if (epoch + 1) % cfg.save_every == 0:
            _save(out_dir / "last.pth", model, ema, cfg, cfg_raw, best_score, epoch)

    return {"best": best_score, "history": history, "ckpt": str(out_dir / "best.pth")}


def _with_ema(model, ema: EMA):
    """复制一份套用 EMA 权重的模型（不污染训练中的模型）。"""
    m = copy.deepcopy(model).eval()
    sd = m.state_dict()
    for k, v in ema.state_dict().items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    m.load_state_dict(sd, strict=False)
    return m


def _spec_from_model(model) -> list[tuple[str, int]]:
    """从模型结构本身推导 ``cls_spec``（``[(字段名, 类别数), ...]``）。

    为什么不"裸导入 ``model`` 模块"取 ``CLS_SPEC``：那种写法依赖当前工作
    目录与 ``sys.path``，一旦导入失败旧实现会**静默**存成 ``[]``。后果是权重
    里明明有 14 个分类头，却没有"哪一头对应哪个字段"的信息，推理侧只能建出
    空头或按位置猜——分类结果静默错位，且不报任何错。

    模型本来就是**按 spec 建出来的**，直接读它的 ``cls_heads`` 与
    ``field_names`` 才是唯一权威来源（顺序天然一致）。
    """
    def _attr(m, name):
        v = getattr(m, name, None)
        if v is None and hasattr(m, "module"):                    # DataParallel 兼容
            v = getattr(m.module, name, None)
        return v

    heads, names = _attr(model, "cls_heads"), _attr(model, "field_names")
    if heads is None or names is None or len(heads) != len(names):
        return []
    return [(str(n), int(h.out_features)) for n, h in zip(names, heads)]


def _save(path: Path, model, ema: EMA, cfg: TrainConfig, cfg_raw: dict,
          best: float, epoch: int) -> None:
    """保存 checkpoint（含重建网络所需的全部元信息）。

    ``cls_spec`` 由 :func:`_spec_from_model` 从模型结构推导——推理侧靠它
    把"头"与"字段含义"对应起来；缺了它就只能按位置猜，顺序一错全盘皆错。
    """
    import torch

    _spec = _spec_from_model(model)
    n_heads = len(getattr(model, "cls_heads", None) or [])
    if n_heads and len(_spec) != n_heads:
        raise RuntimeError(
            f"无法推导 cls_spec（模型有 {n_heads} 个分类头，推导到 {len(_spec)} 个）。"
            f"拒绝保存缺少元信息的权重：推理侧依赖 cls_spec 把'分类头'与'字段'"
            f"对应起来，缺失会导致所有结构化字段静默错位。"
        )
    mc = dict(cfg_raw.get("model") or {})
    torch.save({
        "model": model.state_dict(),
        "model_ema": ema.state_dict(),
        "cls_spec": list(_spec),
        "arch": str(mc.get("arch", "mednext")),
        "model_cfg": {**mc, "in_ch": int(mc.get("in_channels", 4))},
        "thresholds": [0.5, 0.5],                                 # 由全图标定脚本写回
        "train_cfg": {"epochs": cfg.epochs, "lr": cfg.lr, "patch": list(cfg.patch),
                      "seed": cfg.seed, "tag": cfg.tag},
        "global_size": 96,
        "global_size_mm": 192.0,
        "best_score": best,
        "epoch": epoch,
    }, str(path))
    print(f"[train] 已保存 {path}（best={best:.4f}, epoch={epoch}）", flush=True)


def write_history(path: str, history: list[dict]) -> None:
    """训练历史落盘（JSONL，供曲线复盘）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
