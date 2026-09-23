#!/usr/bin/env python
"""核验并合并"人工修补过的异常影像标签表"（如 ``1_abnormal_wzh.xlsx``）回
官方 ``labels/1_abnormal.xlsx``。

为什么需要"核验"：``1_abnormal.xlsx`` 的 ``Label`` **不是"真/假"，而是"来源"**
（``true`` / ``fake`` / ``compositing`` / ``duplicate``）—— 官方 ``validate_data.py``
就是把它当 ``source`` 传给 ``PathResolver.series_path()`` 拼路径的::

    path = resolver.series_path(str(r['AccessionNumber']), str(r['SeriesUid']), str(r['Label']))

于是"标签错"等价于"路径指错目录"，**可以用磁盘机械判定谁对**：按每行 Label 拼
``<根>[/来源]/<检查号>/<序列号>/``，看那里是否真的躺着影像文件。

默认只报告（dry-run），``--apply`` 才写回；写回前自动备份；若补丁的磁盘命中率低于
原表则拒绝写回（``--force`` 可越过）。

用法::

    # ① 只看差别 + 磁盘命中率（不写任何文件）
    python scripts/32_apply_abnormal_patch.py \\
        --base  /2026aicompetition/workspace/dcs/goal1and2/Goal1and2/labels/1_abnormal.xlsx \\
        --patch /2026aicompetition/workspace/dcs/goal1and2/Goal1and2/1_abnormal_wzh.xlsx \\
        --root  /2026aicompetition/datasets/training/annotation

    # ② 确认后再落地（自动备份为 1_abnormal.bak-<时间>.xlsx）
    python scripts/32_apply_abnormal_patch.py --base ... --patch ... --root ... --apply

合并语义：以 ``(检查号, 序列号)`` 为键，**补丁只覆盖 ``Label`` 一列**；补丁里有、
原表没有的键按整行新增。补丁里没有的键保持原值不动（不会"删行"）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.labels import (  # noqa: E402
    LABELS_DIR_ENV, _detect_header, _find_col_in_list, _sheet_rows,
)

ACC_KWS = ("accessionnumber", "accession", "检查号", "patientid")
UID_KWS = ("seriesuid", "series_uid", "序列号")
LABEL_KWS = ("label", "标签")

#: ``Label`` → 该行影像所在子目录（官方 ``paths.py`` 约定；``true``=主目录）
SOURCE_DIRS: dict[str, str] = {
    "true": "", "normal": "", "": "",
    "fake": "fake",
    "compositing": "compositing", "composition": "compositing",
    "duplicate": "duplicate",
}
IMG_EXTS = (".nii.gz", ".nii", ".dcm", ".dicom", ".nrrd", ".mha")
SOURCE_ORDER = ("true", "fake", "compositing", "duplicate")


# --------------------------------------------------------------------------- #
# 读表（不假设第一行是表头，复用仓库已有的表头探测）
# --------------------------------------------------------------------------- #
def load_table(path: str) -> dict:
    """→ ``{header, data, cols, head_row, n_sheets}``；``data`` 为 ``list[list[str]]``。"""
    sheets = _sheet_rows(str(path))
    for rows in sheets:
        if not rows:
            continue
        head = _detect_header(rows)
        if head is None:                                    # 兜底：只要有检查号列
            for i, row in enumerate(rows[:20]):
                if _find_col_in_list([str(c).strip() for c in row], ACC_KWS) is not None:
                    head = i
                    break
        if head is None:
            continue
        header = [str(c).strip() for c in rows[head]]
        i_acc = _find_col_in_list(header, ACC_KWS)
        i_uid = _find_col_in_list(header, UID_KWS)
        i_lab = _find_col_in_list(header, LABEL_KWS)
        if i_acc is None or i_lab is None:
            continue
        data = [list(r) for r in rows[head + 1:] if any(str(c).strip() for c in r)]
        return {"header": header, "data": data, "cols": (i_acc, i_uid, i_lab),
                "head_row": head, "n_sheets": len(sheets)}
    raise SystemExit(f"[32] 读不出表：{path}（前 20 行找不到含检查号列的表头）")


def _cell(row: list, idx: int | None) -> str:
    return str(row[idx]).strip() if idx is not None and idx < len(row) else ""


def entries_of(table: dict) -> tuple[dict, int]:
    """→ ``{(检查号, 序列号): label}`` + 重复键数（后出现的覆盖先出现的）。"""
    i_acc, i_uid, i_lab = table["cols"]
    out: dict[tuple[str, str], str] = {}
    dup = 0
    for row in table["data"]:
        acc = _cell(row, i_acc)
        if not acc:
            continue
        key = (acc, _cell(row, i_uid))
        if key in out:
            dup += 1
        out[key] = _cell(row, i_lab).lower()
    return out, dup


# --------------------------------------------------------------------------- #
# 磁盘核对：Label 说的来源目录里，到底有没有这个序列的影像
# --------------------------------------------------------------------------- #
class DiskCheck:
    def __init__(self, root: str):
        self.root = Path(root)
        self._listing: dict[tuple[str, str], dict[str, str]] = {}

    def series_dir(self, acc: str, uid: str, source: str) -> Path | None:
        sub = SOURCE_DIRS.get(source)
        if sub is None:
            return None
        base = self.root / sub if sub else self.root
        acc_dir = base / acc
        ck = (str(base), acc)
        listing = self._listing.get(ck)
        if listing is None:
            try:                                   # 目录名大小写不严时兜底
                listing = {n.casefold(): n for n in os.listdir(acc_dir)}
            except OSError:
                listing = {}
            self._listing[ck] = listing
        name = listing.get(uid.casefold())
        return acc_dir / name if name else None

    @staticmethod
    def has_image(d: Path | None) -> bool:
        if d is None or not d.is_dir():
            return False
        try:
            return any(n.lower().endswith(IMG_EXTS) for n in os.listdir(d))
        except OSError:
            return False

    def locate(self, acc: str, uid: str, label: str) -> tuple[bool, str | None]:
        """→ ``(按 label 拼路径是否命中, 实际命中的来源/None)``。"""
        if self.has_image(self.series_dir(acc, uid, label)):
            return True, label
        for src in SOURCE_ORDER:
            if src != label and self.has_image(self.series_dir(acc, uid, src)):
                return False, src
        return False, None


def evaluate(entries: dict[tuple[str, str], str], disk: DiskCheck | None) -> dict:
    per = defaultdict(lambda: [0, 0])                       # label → [命中, 总数]
    actual = Counter()
    unknown = 0
    if disk is None:
        return {"per": dict(per), "actual": actual, "unknown": 0, "ok": 0, "n": 0}
    ok_all = 0
    for (acc, uid), lab in entries.items():
        per[lab][1] += 1
        if lab not in SOURCE_DIRS:
            unknown += 1
            continue
        ok, where = disk.locate(acc, uid, lab)
        if ok:
            per[lab][0] += 1
            ok_all += 1
        else:
            actual[where or "(磁盘上找不到)"] += 1
    return {"per": dict(per), "actual": actual, "unknown": unknown,
            "ok": ok_all, "n": len(entries)}


def report(tag: str, res: dict, disk_ok: bool) -> None:
    pct = (100.0 * res["ok"] / res["n"]) if res["n"] else 0.0
    print(f"[{tag}] 行={res['n']}  按 Label 拼路径命中={res['ok']} ({pct:.1f}%)"
          + (f"  未知来源标签={res['unknown']}" if res["unknown"] else ""), flush=True)
    dist = "  ".join(f"{k or '(空)'}:{v[0]}/{v[1]}" for k, v in sorted(res["per"].items()))
    print(f"        Label 分布（命中/总数）：{dist}", flush=True)
    if disk_ok and res["actual"]:
        top = "  ".join(f"{k}:{v}" for k, v in res["actual"].most_common(6))
        print(f"        未命中行的真实位置：{top}", flush=True)


# --------------------------------------------------------------------------- #
def _looks_like_annot(d: Path) -> bool:
    """这一层像不像"病例层"：直接含 fake/compositing/duplicate，或含"检查号/序列号"结构。"""
    if not d.is_dir():
        return False
    try:
        names = os.listdir(d)
    except OSError:
        return False
    if any(n.lower() in ("compositing", "composition", "fake", "duplicate") for n in names):
        return True
    for n in names:
        if n.startswith(".") or n.lower() == "annotation":
            continue
        sub = d / n
        try:
            if sub.is_dir() and any((sub / m).is_dir() for m in os.listdir(sub)):
                return True
        except OSError:
            continue
    return False


def resolve_root(root: str) -> tuple[Path, str]:
    """把用户给的"数据根"归到**病例层**：原样 → ``<root>/annotation``（平台常见）。"""
    base = Path(root).expanduser()
    if _looks_like_annot(base):
        return base, "原样"
    deeper = base / "annotation"
    if _looks_like_annot(deeper):
        return deeper, "自动下钻 annotation/"
    return base, "未识别到典型布局（按原样使用，命中率可能为 0）"


def main() -> int:
    ap = argparse.ArgumentParser(description="核验并合并 1_abnormal 人工补丁")
    ap.add_argument("--base",
                    default=(str(Path(os.environ[LABELS_DIR_ENV]) / "1_abnormal.xlsx")
                             if os.environ.get(LABELS_DIR_ENV) else None),
                    help="现用的 labels/1_abnormal.xlsx（默认 $GLIOMA_LABELS_DIR/1_abnormal.xlsx）")
    ap.add_argument("--patch", required=True, help="人工修补版（如 1_abnormal_wzh.xlsx）")
    ap.add_argument("--out", default=None, help="写回路径（默认就地覆盖 --base）")
    ap.add_argument("--root", default=os.environ.get("DATASET_ROOT")
                    or "/2026aicompetition/datasets/training",
                    help="数据根或病例层，会自动下钻 annotation/（默认 $DATASET_ROOT）")
    ap.add_argument("--apply", action="store_true", help="真的写回（默认只报告）")
    ap.add_argument("--force", action="store_true", help="补丁命中率不升时也写回")
    ap.add_argument("--no-backup", action="store_true", help="写回前不备份")
    args = ap.parse_args()

    if not args.base:
        print("[32] 未给 --base，且环境里没有 $GLIOMA_LABELS_DIR", file=sys.stderr)
        return 2
    base_path, patch_path = Path(args.base), Path(args.patch)
    out_path = Path(args.out) if args.out else base_path
    for p in (base_path, patch_path):
        if not p.is_file():
            print(f"[32] 文件不存在：{p}", file=sys.stderr)
            return 2
    if patch_path.resolve() == base_path.resolve():
        print("[32] --patch 与 --base 是同一个文件，无可合并", file=sys.stderr)
        return 2

    tb, tp = load_table(str(base_path)), load_table(str(patch_path))
    eb, dup_b = entries_of(tb)
    ep, dup_p = entries_of(tp)
    print(f"[32] base  = {base_path}")
    print(f"     表头行={tb['head_row']} 工作表数={tb['n_sheets']} 数据行={len(tb['data'])} "
          f"唯一键={len(eb)}" + (f" 重复键={dup_b}" if dup_b else ""))
    print(f"[32] patch = {patch_path}")
    print(f"     表头行={tp['head_row']} 工作表数={tp['n_sheets']} 数据行={len(tp['data'])} "
          f"唯一键={len(ep)}" + (f" 重复键={dup_p}" if dup_p else ""))
    print(f"     表头 base={tb['header']}")
    print(f"     表头 patch={tp['header']}")

    disk = None
    if args.root:
        layer, how = resolve_root(str(args.root))
        if not layer.is_dir():
            print(f"[32] 警告：{layer} 不存在，跳过磁盘核对", flush=True)
        else:
            if not _looks_like_annot(layer):
                print(f"[32] 警告：{layer} 不像病例层（{how}）→ 命中率可能全 0", flush=True)
            disk = DiskCheck(str(layer))
            print(f"[32] --root = {layer}（{how}）", flush=True)
    else:
        print("[32] 未给 --root 且无 $DATASET_ROOT → 跳过磁盘核对（只比表）", flush=True)

    # ---- 合并（只在内存里）----
    i_acc, i_uid, i_lab = tb["cols"]
    header_b = tb["header"]
    data = [list(r) + [""] * max(0, len(header_b) - len(r)) for r in tb["data"]]
    index: dict[tuple[str, str], list[int]] = {}
    for ri, row in enumerate(data):
        index.setdefault((_cell(row, i_acc), _cell(row, i_uid)), []).append(ri)

    # 补丁原始行（键 → 行），用于"补丁新增键"按**列名**并入
    i_src = {name.strip().lower(): i for i, name in enumerate(tp["header"])}
    prow_of = {(_cell(r, tp["cols"][0]), _cell(r, tp["cols"][1])): r for r in tp["data"]}

    transitions = Counter()
    added = 0
    for key, lab in ep.items():
        rows_here = index.get(key)
        if not rows_here:                                   # 补丁新增的键 → 整行并入
            prow = prow_of.get(key)
            new_row = [""] * len(header_b)
            if prow is not None:
                for j, name in enumerate(header_b):
                    j_src = i_src.get(name.strip().lower())
                    if j_src is not None:
                        new_row[j] = _cell(prow, j_src)
            data.append(new_row)
            index[key] = [len(data) - 1]
            added += 1
            continue
        for ri in rows_here:                                # 重复键 → 该键的每一行都改
            old = _cell(data[ri], i_lab).lower()
            if old != lab:
                transitions[(old, lab)] += 1
                data[ri][i_lab] = lab

    keys_b = set(eb)
    keys_p = set(ep)
    ea = {(_cell(r, i_acc), _cell(r, i_uid)): _cell(r, i_lab).lower() for r in data}

    print()
    print(f"[32] 合并：改 Label={sum(transitions.values())}  新增行={added}  "
          f"补丁未覆盖（保持原值）={len(keys_b - keys_p)}")
    if transitions:
        for (o, n), c in transitions.most_common(12):
            print(f"       {o or '(空)'} → {n or '(空)'} : {c}")
    if keys_p - keys_b:
        print(f"       补丁新增键 {len(keys_p - keys_b)} 个（原表没有）")

    # ---- 磁盘核对：改前 vs 改后 ----
    if disk is not None:
        print()
        res_b = evaluate(eb, disk)
        res_a = evaluate(ea, disk)
        report("改前 base ", res_b, True)
        report("改后 merged", res_a, True)
        pct_b = 100.0 * res_b["ok"] / res_b["n"] if res_b["n"] else 0.0
        pct_a = 100.0 * res_a["ok"] / res_a["n"] if res_a["n"] else 0.0
        verdict = "命中率上升（补丁被磁盘证据支持）" if pct_a > pct_b else \
                  ("持平" if abs(pct_a - pct_b) < 1e-9 else "命中率下降（补丁与磁盘不符！）")
        print(f"[32] 结论：{pct_b:.1f}% → {pct_a:.1f}%  {verdict}", flush=True)
        if res_a["ok"] == 0 and res_b["ok"] == 0:
            print("[32] 两表命中率都是 0 → --root 很可能指错了层：它应指向**直接含"
                  "检查号目录与 fake/compositing/duplicate 的那层**（通常是 <数据根>/annotation）",
                  file=sys.stderr)
            if not args.force:
                return 4
        if pct_a < pct_b - 1e-9 and not args.force:
            print("[32] 补丁命中率更低 → 拒绝写回（确认无误可加 --force）", file=sys.stderr)
            return 3

    if not args.apply:
        print("[32] dry-run：未写任何文件。确认无误后加 --apply 落地。")
        return 0

    if not transitions and not added:
        print("[32] 无差异：补丁与 base 一致（已是合并后状态），不写回。")
        return 0

    # ---- 落地 ----
    if tb["head_row"] != 0 or tb["n_sheets"] > 1:
        print(f"[32] 注意：原表有前置行/多工作表（head_row={tb['head_row']} "
              f"sheets={tb['n_sheets']}），写回会只保留单表数据（列名与数据不丢）。",
              flush=True)
    if out_path.exists() and not args.no_backup:
        bak = out_path.with_name(f"{out_path.stem}.bak-{datetime.now():%Y%m%d-%H%M%S}"
                                 f"{out_path.suffix}")
        shutil.copy2(out_path, bak)
        print(f"[32] 已备份 → {bak.name}")

    import pandas as pd
    df = pd.DataFrame(data, columns=header_b)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(out_path, index=False)
    print(f"[32] 已写出 → {out_path}（行={len(df)}）")
    print("[32] 下一步：export GLIOMA_LABELS_DIR=<labels 目录> 后重跑 scripts/01_probe.sh，"
          "核对 abnormal_label_counts。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
