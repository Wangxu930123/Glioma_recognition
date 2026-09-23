#!/usr/bin/env python
"""把标注/金标准表**整个摊开**看（不再猜列名）。

什么时候用：探针报 `label_field_counts: {}`、`labels_hint` 说"找到表但一行都没解析出来"。
与其逐个猜列名，不如直接把表的真实内容打印出来：工作表、行列数、前若干行、
以及"哪一列能被识别成检查号、命中多少行"。

用法::

    # ① 不带参数：把数据根（及其上级）下所有候选表都摊开看
    python scripts/30_inspect_table.py

    # ② 指定文件
    python scripts/30_inspect_table.py "/path/脑胶质瘤标注结果-训练集.xlsx"

    # ③ 看更多行/列
    python scripts/30_inspect_table.py --rows 12 --cols 16

判定逻辑（**不依赖列名**）：拿磁盘上真实的检查号目录名，逐列比对取值，
命中率最高的一列就是检查号列 —— 列名叫 `编号`/`AccessionNumber`/乱码都不影响。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.labels import (dump_table, find_structured_tables,     # noqa: E402
                             read_structured_table, structured_from_row,
                             _best_id_column_by_values, _id_key, _sheet_rows)


def _known_ids(root: Path) -> set[str]:
    """磁盘上真实存在的检查号（一级子目录，排除 annotation）。"""
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir()
            if p.is_dir() and p.name.lower() != "annotation"}


def _report(path: str, known: set[str], rows: int, cols: int) -> None:
    print("=" * 78)
    print(dump_table(path, max_rows=rows, max_cols=cols))
    print("-" * 78)

    # 逐 sheet 说明"哪一列被认成检查号、命中多少行"
    for idx, sheet in enumerate(_sheet_rows(path)):
        if not sheet:
            continue
        width = max(len(r) for r in sheet)
        best = _best_id_column_by_values(sheet, {_id_key(k) for k in known}) if known else None
        if best:
            col, first = best
            hits = sum(1 for r in sheet
                       if col < len(r) and _id_key(r[col]) in {_id_key(k) for k in known})
            print(f"  sheet{idx}: 检查号列 = 第 {col} 列"
                  f"（表头取值 {str(sheet[first - 1][col])[:24]!r}）"
                  f" …命中 {hits} 行 / 共 {len(sheet)} 行")
        else:
            head = [str(c)[:18] for c in sheet[min(1, len(sheet) - 1)][:8]]
            print(f"  sheet{idx}: ⚠️ 没有哪一列的取值能对上磁盘上的检查号"
                  f"（width={width}，前几列表头={head}）")
            print("           → 要么这表不是字段金标准，要么检查号被改过名")

    table = read_structured_table(path, known_ids=known)
    if table:
        sample = next(iter(table.values()))
        fields = structured_from_row(sample)
        print(f"  解析结果：{len(table)} 个键，样例可映射字段 {len(fields)} 个 {sorted(fields)[:6]}")
    else:
        print("  解析结果：0 行（已尝试按取值/按列名/跳过标题行/遍历全部 sheet）")


def main() -> int:
    ap = argparse.ArgumentParser(description="摊开看标注/金标准表")
    ap.add_argument("path", nargs="?", default=None, help="表文件；省略则看数据根下的候选表")
    ap.add_argument("--root", default=os.environ.get("DATASET_ROOT"),
                    help="数据根（默认取 DATASET_ROOT）")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=12)
    a = ap.parse_args()

    root = Path(a.root).expanduser() if a.root else None
    known = _known_ids(root) if root else set()
    if root:
        print(f"数据根：{root}")
        print(f"磁盘上的检查号：{len(known)} 个"
              + (f"，例如 {sorted(known)[:2]}" if known else "（取不到，将只按列名识别）"))

    targets: list[str] = []
    if a.path:
        targets = [a.path]
    elif root:
        targets = find_structured_tables(str(root))
        if not targets:
            print("⚠️ 数据根及其上级 1~2 层都没找到 csv/xlsx；用 --root 指定数据根，"
                  "或直接传入文件路径。")
            return 1
    else:
        print("请给 --root 或直接传入表文件路径")
        return 2

    for t in targets:
        _report(t, known, a.rows, a.cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
