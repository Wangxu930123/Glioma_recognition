"""pytest 内存探针：逐测试报告 RSS 增量与峰值（OOM 排障用）。

**为什么需要它**：判断"OOM 是不是测试套件自己造成的"，靠猜没用 —— 实测一次就知道。
本仓实测结果（66 个测试）：峰值 358 MB，**没有任何单个测试增量超过 20 MB**，
即测试套件本身不会 OOM 容器；OOM 的成因在**权重目录内容**或**推理期峰值**。

用法::

    python -m pytest tests -p scripts.memprobe -q
    python -m pytest tests -p scripts.memprobe -q --tb=no

输出：每个测试结束后的 RSS 增量；末尾给出增量 top 列表与峰值。
"""
from __future__ import annotations

import gc
import os

try:
    import psutil

    _PROC = psutil.Process(os.getpid())
except Exception:                                                 # noqa: BLE001
    _PROC = None

_rows: list[tuple[str, float, float]] = []
_prev = [0.0]
_phase_prev = [0.0]


def _rss() -> float:
    return _PROC.memory_info().rss if _PROC is not None else 0.0


def pytest_sessionstart(session) -> None:                         # noqa: ANN001
    gc.collect()
    _prev[0] = _rss()
    _phase_prev[0] = _prev[0]
    if _PROC is None:
        print("\n[memprobe] !! 没有 psutil → 无法测量（pip install psutil）")
    else:
        print(f"\n[memprobe] 起始 RSS = {_prev[0] / 1e6:.0f} MB")


def pytest_runtest_teardown(item, nextitem) -> None:              # noqa: ANN001
    if _PROC is None:
        return
    gc.collect()
    cur = _rss()
    _rows.append((item.nodeid, cur, cur - _prev[0]))
    _prev[0] = cur
    d = cur - _phase_prev[0]
    if abs(d) > 80e6:                                             # 单测跳变 > 80MB
        print(f"\n[memprobe] !! {item.nodeid}  RSS {d / 1e6:+.0f} MB → {cur / 1e6:.0f} MB")
    _phase_prev[0] = cur


def pytest_sessionfinish(session, exitstatus) -> None:            # noqa: ANN001
    if not _rows:
        return
    print("\n" + "=" * 92)
    print("逐测试 RSS 增量（按增量降序；只列 > 20MB 或累计 > 1200MB 的）")
    print("=" * 92)
    shown = 0
    for nid, cur, d in sorted(_rows, key=lambda r: -r[2]):
        if d > 20e6 or cur > 1.2e9:
            print(f"  +{d / 1e6:8.1f} MB   累计 {cur / 1e6:8.1f} MB   {nid}")
            shown += 1
    if not shown:
        print("  （没有任何单个测试增量超过 20MB → 测试套件不是 OOM 源）")
    peak = max((r[1] for r in _rows), default=0.0)
    print("-" * 92)
    print(f"测试数={len(_rows)}  峰值 RSS={peak / 1e6:.0f} MB  "
          f"结束 RSS={_rss() / 1e6:.0f} MB")
    print("=" * 92)
