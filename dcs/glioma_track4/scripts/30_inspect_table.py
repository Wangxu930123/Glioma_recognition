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

判定逻辑与解析器**同源**：逐 sheet 调 `labels._sheet_plan()`，报出解析器实际认下的
"表头在第几行 / 行键是第几列 / 实际级别"，再补一句命中多少行。所以这里显示的结论
就是解析器会做的事 —— 工具与解析器各写一套表头识别必然互相矛盾（曾把表头在第 1 行的
序列级 sheet 报成"认不出行键列"，好表说成坏表）。

检查号列的判定**不依赖列名**：拿磁盘上真实的检查号目录名逐列比对取值，命中率最高的
一列就是检查号列 —— 列名叫 `AccessionNumber`（数据集各表的规范拼写）/历史排版里的拼写
变体（`Accessionumber`、`AccessioNumber`）/乱码都不影响。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.labels import (LEVEL_NESTED_KEY, SERIES_TYPE_TABLE,    # noqa: E402
                             dump_table, find_structured_tables, read_series_types,
                             read_structured_table, structured_from_row,
                             _find_case_column, _id_key, _sheet_frames, _sheet_level,
                             _sheet_plan)


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

    if os.path.basename(path) == SERIES_TYPE_TABLE:
        # 序列类型表**不是**字段金标准：行键是 (检查号, 序列号)、取值是模态。
        # 套字段金标准那套逻辑去读它必然报"0 行 / 认不出检查号"，把好表说成坏表 ——
        # 换用真正读它的那个函数，直接给"读到几条 + 取值分布 + 命中了多少病例"。
        types = read_series_types(os.path.dirname(path))
        print(f"  解析结果：{len(types)} 条序列类型（(检查号, 序列号) → 模态），"
              f"取值分布 {dict(Counter(types.values()))}")
        if known:
            hit = {_id_key(k[0]) for k in types} & {_id_key(k) for k in known}
            print(f"  命中磁盘上的病例：{len(hit)} / {len(known)} 例")
        return

    # 逐 sheet 说明"这张表是什么级别、哪一列被认成行键、命中多少行"。
    #
    # ⚠️ 这里**直接复用解析器的 `_sheet_plan`**，不自己另写一套表头识别：
    # 另写一套必然与解析器不一致（本工具曾写死"表头在第 2 行"，遇到表头在第 1 行的
    # 序列级 / ROI 级 sheet 就把**数据行当表头**，报成"认不出行键列"——
    # 好表被说成坏表，而解析器那边其实好好的）。级别与行键必须与解析器同源。
    known_keys = {_id_key(k) for k in known}
    for idx, (name, sheet) in enumerate(_sheet_frames(path)):
        if not sheet:
            continue
        named = _sheet_level(name)
        plan = _sheet_plan(sheet, named, known_keys or None)
        if plan is None:
            preview = [str(c).strip()[:16] for c in sheet[min(1, len(sheet) - 1)][:6]]
            print(f"  sheet{idx} {name!r}（表名判级={named}）: ⚠️ 定不出表头 / 行键列"
                  f"（前两行之一={preview}）")
            print("           → 用 dump_table 的表格输出对照一下：行键列名是否在候选里")
            continue
        header, data_start, key_col, level = plan
        filled = sum(1 for r in sheet[data_start:]
                     if key_col < len(r) and str(r[key_col]).strip())
        line = (f"  sheet{idx} {name!r}（{level}"
                + ("" if level == named else f"（表名判为 {named}，按表头列改判）") + "）："
                + f"表头在第 {data_start} 行、行键 = 第 {key_col} 列"
                + f"（表头 {header[key_col][:24]!r}）…非空 {filled} 行 / 共 {len(sheet)} 行")
        if level == "case" and known_keys:
            hits = sum(1 for r in sheet[data_start:]
                       if key_col < len(r) and _id_key(r[key_col]) in known_keys)
            line += f"，命中磁盘检查号 {hits} 行"
        if level in LEVEL_NESTED_KEY:                         # 序列级 / ROI 级：靠检查号列挂病例
            case_col, trusted = _find_case_column(sheet, header, key_col,
                                                  known_keys or None)
            line += (f"，检查号列 = 第 {case_col} 列"
                     f"（{'按取值比对' if trusted else '按列名认'}）"
                     if case_col is not None else "，⚠️ 认不出检查号列 → 子行挂不到病例")
        print(line)

    table = read_structured_table(path, known_ids=known)
    if table:
        # 同一行会登记多个别名键（原值 / 去前导零 / 大小写折叠）→ 按**唯一记录**计数
        cases = {id(v): v for v in table.values()}.values()
        sample = next(iter(cases))
        fields = structured_from_row(sample)
        print(f"  解析结果：{len(cases)} 例（{len(table)} 个键），"
              f"样例可映射字段 {len(fields)} 个 {sorted(fields)[:6]}")
        for level, key in LEVEL_NESTED_KEY.items():
            n = sum(len(case.get(key) or []) for case in cases)
            if n:
                print(f"  {level} 级子行：{n} 行（挂在病例记录的 {key!r} 下，不参与字段映射）")
    else:
        # ⚠️ 表里只有序列级 / ROI 级的行（没有"检查级别"的行来立病例）时，这些行**只能挂到
        # 已存在的病例上**；没给数据根（取不到磁盘检查号）就一条都挂不上。
        # 此时统一报"0 行（已尝试…）"等于把好表说成坏表 —— 上面逐 sheet 明明认出了行键。
        orphan = 0
        for _n, _s in _sheet_frames(path):
            _p = _sheet_plan(_s, _sheet_level(_n), known_keys or None)
            if not _p or _p[3] not in LEVEL_NESTED_KEY:
                continue
            _ds, _kc = _p[1], _p[2]
            orphan += sum(1 for _r in _s[_ds:]
                          if _kc < len(_r) and str(_r[_kc]).strip())
        if orphan:
            print(f"  解析结果：0 例 —— 本表只有 {orphan} 行**序列级/ROI 级**数据，"
                  f"没有『检查级别』的行来立病例；\n"
                  f"    这类行只能挂到**已存在的病例**上（防「按猜出来的检查号列凭空造病例」），"
                  f"把数据根指到含检查号目录的那一层（`--root <阶段目录>`）即可对上。")
        else:
            print("  解析结果：0 行（已尝试按工作表名分级 / 按取值 / 按列名 / 跳过标题行）")


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
