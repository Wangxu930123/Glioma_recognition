#!/usr/bin/env python
"""用一张数据信息表（``SeriesType.xlsx``）造**官方布局**的本地小数据集（无需真实数据）。

**为什么不直接用 99_smoke_test.py**：它造的是**模态名目录**（``t1c_0000``）——模态直接从
目录名就猜出来了，而线上唯一会炸的形态（序列目录名是 UID、模态只能来自 ``SeriesType.xlsx``）
在它那儿根本不出现。本脚本按表里的**真实检查号/序列 UID** 建目录，于是 01→03 走的是与线上
完全相同的那条链（也就能复现"表没接上 → 一例都挑不出模态"）。

生成结构（与平台契约一致）::

    <out>/annotation/SeriesType.xlsx                       ← 表与病例目录同层
    <out>/annotation/<32位检查号>/<序列UID>.nii.gz                  ← 影像（与掩膜**同目录**）
    <out>/annotation/<32位检查号>/<序列UID>_肿瘤瘤体_1_mask.nii.gz   ← 该病例有 T1CE 才有（core）
    <out>/annotation/<32位检查号>/<序列UID>_瘤体_2_mask.nii.gz       ← 有 T2-Flair 才写（peri）
    <out>/annotation/<32位检查号>/<序列UID>_水肿_3_mask.nii.gz
    <out>/annotation/structured.csv                        ← 结构化字段金标准（病例级）

表里的 ``其他`` 序列**照原样生成**：它是**权威排除**，探针应把它放进 ``images["other"]``
并标 ``declared_other``，而不是丢给体素模型猜。

生成完会**立刻用探针自己的收集函数自检**（见 :func:`_verify_roles`）：
掩膜与影像同目录、掩膜的模态只能靠"从文件名剥出裸 UID 再查表"拿到，而
``瘤体`` 的角色**依赖模态**（FLAIR/T2→peri、T1/T1CE→core）—— 这条链断了
会静默落成 core，只有断言 ``瘤体``（FLAIR）= peri 才抓得住。

用法::

    python scripts/34_make_local_dataset.py --table data/SeriesType.xlsx --out data/local_uid --force
    python scripts/34_make_local_dataset.py --shape 24 --limit 8 --force     # 更快更小
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 官方掩膜命名：``<序列UID>_<RoiName>_<RoiNumber>_mask.nii.gz``（掩膜与影像**同目录**）。
#: ``RoiName`` / ``RoiNumber`` 取自 ``脑胶质瘤标注结果-训练集.xlsx`` 的 ``ROI级别``
#: sheet 的 **AC / AD 列**。
#:
#: 搭配必须与规范一致，否则本地演练测不出真问题：
#: 任务A(core) = T1CE 上的 ``肿瘤瘤体``；任务B(peri) = FLAIR/T2 上的 ``瘤体 ∪ 水肿``。
#: 这里 FLAIR 上故意写 ``瘤体``（**不是** ``肿瘤瘤体``）：它的角色**只能**靠
#: "掩膜名 → 剥出裸 UID → 查类型表得模态"这条链判成 peri —— 链断了它会静默落成
#: core，而演练照常跑完、指标只是偏低。
CORE_ROI = ("肿瘤瘤体", 1)
PERI_ROIS = (("瘤体", 2), ("水肿", 3))

#: 各模态的病灶对比度增益（核心, 周边）
_BOOST = {"t1c": (1.8, 1.0), "flair": (0.6, 1.4), "t2": (0.8, 1.2), "t1": (0.3, 0.2)}


def read_table(path: Path) -> list[tuple[str, str, str]]:
    """读 ``SeriesType.xlsx`` → ``[(检查号, 序列UID, 类型)]``（**原文**，要用作目录名）。

    列**按列名定位**、不写死列序：表头拼写改版时列序会变，写死列序会把检查号当序列 UID 用。
    """
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = [list(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
    finally:
        wb.close()
    if not rows:
        raise SystemExit(f"{path} 是空表")
    head = [str(c or "").strip().lower() for c in rows[0]]

    def col(*keys: str) -> int:
        for i, h in enumerate(head):
            if any(k in h for k in keys):
                return i
        raise SystemExit(f"{path} 表头里找不到 {keys}，实际表头={rows[0]}")

    ia, iu, it = (col("accessionnumber", "accession"), col("seriesuid", "studyuid"),
                  col("seriestype", "serieslabel"))
    out = []
    for r in rows[1:]:
        if len(r) <= max(ia, iu, it):
            continue
        acc, uid, typ = (str(r[ia] or "").strip(), str(r[iu] or "").strip(),
                         str(r[it] or "").strip())
        if acc and uid:
            out.append((acc, uid, typ))
    return out


def _mask_name(uid: str, roi: tuple) -> str:
    """``(<序列UID>, (RoiName, RoiNumber))`` → 官方掩膜文件名。"""
    return f"{uid}_{roi[0]}_{roi[1]}_mask.nii.gz"


def _write_series(cdir: Path, uid: str, typ: str, shape: tuple, seed: int) -> None:
    """写一条序列的 NIfTI（脑 + 病灶，按模态给不同对比度）及该模态能承载的掩膜。

    影像与掩膜**同目录**（官方扁平布局），掩膜用官方命名（见 :data:`CORE_ROI`）。
    """
    import nibabel as nib
    from src.data.labels import guess_modality

    rng = np.random.default_rng(seed)
    aff = np.diag([1.0, 1.0, 2.0, 1.0])
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    c = (shape[0] // 2 + int(rng.integers(-2, 3)), shape[1] // 2, shape[2] // 2)
    d = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2).astype(np.float32)
    brain, core, peri = d <= min(shape) // 2 - 2, d <= 4, (d <= 7) & (d > 4)
    vol = brain * (0.4 + 0.05 * rng.standard_normal(shape).astype(np.float32))
    b_core, b_peri = _BOOST.get(guess_modality(typ) or "", (0.5, 0.5))
    vol[core] += b_core
    vol[peri] += b_peri

    cdir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol.astype(np.float32), aff), str(cdir / f"{uid}.nii.gz"))
    mod = guess_modality(typ)
    if mod == "t1c":
        nib.save(nib.Nifti1Image(core.astype(np.uint8), aff),
                 str(cdir / _mask_name(uid, CORE_ROI)))
    elif mod == "flair":                       # 总异常区 = 瘤体 ∪ 水肿
        nib.save(nib.Nifti1Image(peri.astype(np.uint8), aff),
                 str(cdir / _mask_name(uid, PERI_ROIS[0])))
        nib.save(nib.Nifti1Image((peri & ~core).astype(np.uint8), aff),
                 str(cdir / _mask_name(uid, PERI_ROIS[1])))


def _verify_roles(ann: Path, expect: dict, limit: int) -> list[str]:
    """用**探针自己的** ``scan_real`` 验掩膜角色与模态（不另写一套判据）。

    ``expect``: ``{(检查号, 序列UID): (模态, 序列类型, 期望角色集合)}``。
    返回问题清单（空 = 全对）。必须验的三件事：

    1. 影像进了对应输入通道，且 ``series_uid`` 是**裸 UID**；
    2. 掩膜角色符合规范 —— FLAIR 上的 ``瘤体`` → **peri** 是"掩膜名剥 UID → 查表得模态"
       这条链的探针；
    3. 掩膜所在序列的模态解析出来了：否则 ``瘤体`` 只能按 core 兜底，**不报错**。
    """
    from src.data.probe import scan_real

    cases = {c["accession"]: c for c in scan_real(str(ann), limit_cases=limit)}
    bad: list[str] = []
    seen_mod: set[tuple[str, str]] = set()          # 同一模态多条序列时 images 只留第一条
    for (acc, uid), (mod, typ, want) in expect.items():
        c = cases.get(acc)
        if c is None:
            bad.append(f"{acc}: 探针一例都没认出来（影像与掩膜都没进清单）")
            continue
        meta = c["images"].get(mod)
        if (acc, mod) not in seen_mod:
            seen_mod.add((acc, mod))
            if not meta:
                bad.append(f"{acc}/{uid}: {mod} 未进输入通道（images={sorted(c['images'])}）")
            elif meta.get("series_uid") != uid:
                bad.append(f"{acc}/{uid}: series_uid 解析成 {meta.get('series_uid')!r}（应为裸 UID）")
        got = {role for role, e in (c["masks"] or {}).items()
               for m in (e.get("metas") or []) if m.get("series_uid") == uid}
        if want and not want <= got:
            bad.append(f"{acc}/{uid}（{typ}）掩膜角色 {sorted(got) or '∅'} ⊉ {sorted(want)}")
        if got and not want:
            bad.append(f"{acc}/{uid}（{typ}）不该有掩膜，实得 {sorted(got)}")
    return bad


def main() -> int:
    from src.data.labels import guess_modality, is_explicit_other

    ap = argparse.ArgumentParser(description="用数据信息表造官方布局的本地小数据集")
    ap.add_argument("--table", default=str(ROOT / "data" / "SeriesType.xlsx"))
    ap.add_argument("--out", default=str(ROOT / "data" / "local_uid_dataset"))
    ap.add_argument("--shape", type=int, default=32, help="体素边长（默认 32，越小越快）")
    ap.add_argument("--limit", type=int, default=0, help="最多生成多少病例（0=全部）")
    ap.add_argument("--drop-empty", action="store_true",
                    help="剔除只有 `其他` 序列的病例。**默认保留**（全放开口径：这类病例"
                         "4 个通道全零 + 借几何照训），剔掉就演练不到那条路径")
    ap.add_argument("--verify", type=int, default=8,
                    help="生成后用探针抽验前 N 例的掩膜角色/模态（0=不验）")
    ap.add_argument("--force", action="store_true", help="输出目录已存在时先删除")
    a = ap.parse_args()

    table, out = Path(a.table), Path(a.out)
    if not table.is_file():
        raise SystemExit(f"找不到数据信息表：{table}（用 --table 指定）")
    if out.exists():
        if not a.force:
            raise SystemExit(f"{out} 已存在；确认重建就加 --force（会先删掉该目录）")
        shutil.rmtree(out)
    ann = out / "annotation"
    ann.mkdir(parents=True)

    rows = read_table(table)
    by_acc: dict[str, list[tuple[str, str]]] = {}
    for acc, uid, typ in rows:
        by_acc.setdefault(acc, []).append((uid, typ))
    accs = sorted(by_acc)

    # 只有 `其他` 序列的病例**一条真输入通道都没有**（探针的 `cases_dropped_no_series`
    # 管的是"一个序列目录都没有"，拦不到这类）。**默认保留**：全放开口径下它们照常进
    # 训练（4 通道全零 + 借该例几何），这条路径必须被演练到 —— 否则真数据上第一次
    # 遇到才知道。想复现"老行为"用 `--drop-empty`。
    empty = [x for x in accs
             if not any(guess_modality(t) and not is_explicit_other(t) for _, t in by_acc[x])]
    if a.drop_empty:
        accs = [x for x in accs if x not in set(empty)]
    accs = accs[: a.limit or None]

    shape = (a.shape, a.shape, max(8, a.shape * 3 // 4))
    n_core = n_peri = n_other = 0
    expect: dict[tuple[str, str], tuple[str, str, set]] = {}
    for i, acc in enumerate(accs):
        for j, (uid, typ) in enumerate(by_acc[acc]):
            _write_series(ann / acc, uid, typ, shape, seed=1000 * i + j)
            mod = guess_modality(typ)
            if is_explicit_other(typ):
                n_other += 1
            elif mod:                              # 只有这些行才有掩膜，也是自检的对象
                n_core += int(mod == "t1c")
                n_peri += int(mod == "flair")
                expect[(acc, uid)] = (
                    mod, typ, {"t1c": {"core"}, "flair": {"peri"}}.get(mod, set()))

    shutil.copy2(table, ann / table.name)      # 表与病例目录**同层**（平台契约）

    # 结构化字段金标准（病例级）。真实数据是 `脑胶质瘤标注结果-训练集.xlsx`（3 张工作表），
    # 本地写 csv 即可 —— 探针的 find_structured_tables 同样认它。
    with open(ann / "structured.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["AccessionNumber", "病理结果", "WHO分级", "location_of_lesion"])
        for i, acc in enumerate(accs):
            w.writerow([acc, "脑胶质瘤4级" if i % 3 == 0 else "脑胶质瘤2级",
                        4 if i % 3 == 0 else 2, "右侧额叶" if i % 2 == 0 else "左侧颞叶"])

    print(f"[34] 表：{table}（{len(rows)} 条 / {len(by_acc)} 个检查号）→ {out}")
    print(f"[34] 病例 {len(accs)} 例（体素 {shape}）")
    print(f"[34]   core 掩膜 {n_core} 个（只有 T1CE 行才有）｜"
          f"peri 掩膜 {n_peri} 个（只有 T2-Flair 行才有）")
    print(f"[34]   `其他` 序列 {n_other} 条（权威排除）｜只有 `其他` 的病例 {len(empty)} 个"
          + ("（已按 `--drop-empty` 剔除）" if a.drop_empty else
             "（**已保留**：4 通道全零 + 借几何照训，正是要演练的那条路径）"))
    if n_core == 0:
        print("[34] ⚠️ 表里一条 T1CE 都没有 → 没有 core 掩膜，分割只剩 peri 一路监督。")

    # 生成即自检：**用探针自己的判据**。掩膜与影像同目录，掩膜所在序列的模态只能靠
    # "从文件名剥出裸 UID 再查类型表"拿到，而 `瘤体` 的角色**依赖模态** —— 链断了会
    # 静默落成 core（掩膜进错任务空间），所以这里必须断言，不能只看"跑通了"。
    if a.verify:
        _head = set(accs[: a.verify])
        _n_v = len(_head)
        _bad = _verify_roles(ann, {k: v for k, v in expect.items() if k[0] in _head}, _n_v)
        if _bad:
            print(f"[34] ✗ 自检失败 {len(_bad)} 条 —— 掩膜角色/模态与规范不符，"
                  f"这份假数据不可信（`瘤体`（FLAIR）本应为 peri）：", flush=True)
            for _b in _bad[:10]:
                print(f"[34]   - {_b}", flush=True)
            return 2
        print(f"[34] 自检：探针 `scan_real` 抽验前 {_n_v} 例的角色/模态 → PASS"
              f"（含 FLAIR 上的 `瘤体` → peri，即「掩膜名剥出裸 UID 再查表」这条链）",
              flush=True)

        # ★ 全放开路径自检：整例只有 `其他` 序列的病例**必须能建出体数据**
        #   （4 通道全零 + 借该例任意一路影像的几何），而不是抛错。
        #   这条路径原先的表现是"训练跑到一半、在 DataLoader worker 里随机崩"，
        #   所以必须在**造数据**这一步就被演练到 —— 否则真数据上第一次遇到才知道。
        # 注意**不能**只抽前 `--verify` 例：这类病例散布在清单里，抽样很可能一个都不含
        # （实测：3 例全在 8 例之外 → 自检静默通过 = 等于没检）。它们本来就少，全查。
        _want = set(empty) & set(accs)
        if _want:
            from src.data.dataset import build_case_volume          # noqa: PLC0415
            from src.data.probe import scan_real                    # noqa: PLC0415
            from src.utils.config import load_config                # noqa: PLC0415

            _pre = load_config("preprocess.yaml")
            _fails: list[str] = []
            for _c in scan_real(str(ann), limit_cases=10 ** 6):
                if _c["accession"] not in _want:
                    continue
                if not _c.get("no_input_channel"):
                    _fails.append(f"{_c['accession']}: 探针没标 `no_input_channel`")
                    continue
                try:
                    _v, _aff, _m = build_case_volume(_c, _pre)
                    if _v.shape[0] != len(_pre["channels"]) or _v.any():
                        _fails.append(f"{_c['accession']}: 通道数={_v.shape[0]}"
                                      f" 应为全零却非零={bool(_v.any())}")
                except Exception as _exc:                           # noqa: BLE001
                    _fails.append(f"{_c['accession']}: {type(_exc).__name__}: {_exc}")
            if _fails:
                print(f"[34] ✗ 全放开自检失败：只有 `其他` 的病例建不出体数据："
                      f"{_fails[:3]}", flush=True)
                return 2
            print(f"[34] 自检：全放开 → {len(_want)} 例「只有 `其他`」的病例建出"
                  f"**全零 {len(_pre['channels'])} 通道**体数据（借该例几何），不抛错",
                  flush=True)
    print(f"\n[34] 下一步：\n"
          f'  export DATASET_ROOT="{ann.as_posix()}"\n'
          f"  rm -f data/manifest.json data/folds.json\n"
          f"  bash scripts/01_probe.sh && bash scripts/02_build_dataset.sh\n"
          f"  bash scripts/03_train.sh 2 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
