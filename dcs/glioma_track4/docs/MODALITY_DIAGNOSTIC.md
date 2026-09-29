# 模态识别 1 分钟诊断（上游问题定位）

> 目的：一条命令定位"**为什么挑不出序列**"，并判断体素兜底能不能用。
> 三个诊断合计约 1 分钟（② 较慢，最多 40 例）。
>
> 关联文档：
> - [`INFERENCE_EVAL_AND_SUBMIT_GUIDE.md`](INFERENCE_EVAL_AND_SUBMIT_GUIDE.md) —— 推理/测评/提交全链路
> - [`MIGRATE_AND_RESUME_GUIDE.md`](MIGRATE_AND_RESUME_GUIDE.md) —— §8 故障表 ⑩~⑬（权重真伪/KeyError）
> - [`OOM_AND_RESUME_GUIDE.md`](OOM_AND_RESUME_GUIDE.md) —— 内存与断点续训
>
> 所有代码块都可整段粘贴。**先看 §0 的因果链，再决定跑哪几个诊断。**

---

## 0. 先看懂因果链（30 秒）

模态识别有**三条腿**，依次尝试；三条都失败 → Goal5 输入通道全零 → **该例分割必然 0 分**：

```text
① 关键词（data/series_selector.guess_modality）      看序列描述里有没有 t1c/flair/t2
② 查官方表（data/modality_fallback）                  按 (检查号,序列号) 或 SeriesUid 回贴模态
③ 体素统计判别（data/voxel_modality）                只看影像体素，不依赖任何外部文件
```

对应的报错/日志：

| 日志/报错 | 卡在哪 | 看哪个诊断 |
|---|---|---|
| `[selector][模态回退] 挑不出序列，序列表在 .../SeriesType.xlsx 但按 (检查号,序列号) / UID 都匹配不到` | ② 匹配上了**表**，但 UID 键对不上 | **§1c**（表已找到 → 跳到 §1c） |
| `[selector][模态回退] study '...' 原描述认不出模态，已按官方表重贴：{...: 其他, ...}` | ② 匹配上了，但表值是 `其他` | **§1 + §2** |
| `ValueError: study ... 没有任何可用影像` | 三条腿全失败 | **§1 + §2** |
| `KeyError: unknown series UID` | 掩膜绑定了不存在的序列 | 已修（见 MIGRATE 文档 §8） |
| `ValueError: core and flair masks target the same series but differ` | 两路退化到同一序列 | 已修（同上） |
| 推理跑得完但 `core_voxels=0` | 通道全零 → 预测为空 | **§3** |

---

## 1. 诊断① · 官方表读取与取值分布（约 10 秒）

**回答两个问题**：表读到了吗？`其他` 是真的还是**读错列**？

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 - <<'PY'
import sys
sys.path.insert(0, '.')
from pathlib import Path
from collections import Counter
from openpyxl import load_workbook
from data.modality_fallback import _read_rows, _table_in, find_label_table, _norm

ROOT = Path('/2026aicompetition/datasets/verification')
SUB = ROOT / 'original' if (ROOT / 'original').is_dir() else ROOT
print("数据目录   :", SUB)
if not SUB.is_dir():
    raise SystemExit(f"✗ 目录不存在：{SUB}")

accs = sorted(p for p in SUB.iterdir() if p.is_dir())
print("检查号目录数:", len(accs), "| 样例:", [a.name for a in accs[:3]])

# 定位表（数据优先：序列文件所在目录及其上溯 4 层）
sources = [f for a in accs[:2] for f in a.glob('*.nii.gz')]
table = _table_in(SUB) or find_label_table(sources or [SUB])
print("定位到的表 :", table)

print("\n--- 前 3 行原文（看清表头与取值；表头决定 _read_rows 认哪一列）---")
wb = load_workbook(table, read_only=True, data_only=True)
for si, ws in enumerate(wb.worksheets):
    print(f"  [sheet {si}] {ws.title}")
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print("     ", row)
        if i >= 2:
            break
wb.close()

rows = _read_rows(table)
print("\n读到的条数:", len(rows))
print("取值分布 top12:", Counter(rows.values()).most_common(12))
print("前 5 条:")
for k, v in list(rows.items())[:5]:
    print("   ", k, "->", v)

# ---- 与磁盘对齐：文件名 stem / 检查号 能否命中表 ----
print("\n--- 磁盘侧样本 ---")
for a in accs[:3]:
    stems = [f.name[:-len('.nii.gz')] for f in sorted(a.glob('*.nii.gz'))]
    print(f"  {a.name}: {stems[:6]}{' …' if len(stems) > 6 else ''}")

tbl_uids = {u for (_a, u) in rows}
tbl_accs = {a for (a, _u) in rows}
hit_uid = tot = 0
for a in accs[:20]:
    for f in a.glob('*.nii.gz'):
        tot += 1
        hit_uid += (f.name[:-len('.nii.gz')] in tbl_uids)
hit_acc = sum(1 for a in accs[:20] if _norm(a.name) in tbl_accs)
print(f"\n前 20 例共 {tot} 个序列：")
print(f"  文件名 stem 命中「表 UID」   : {hit_uid}/{tot} = {hit_uid/max(1,tot):.0%}")
print(f"  检查号目录命中「表 检查号」  : {hit_acc}/20 = {hit_acc/20:.0%}")
print(f"  表的唯一 UID 数 = {len(tbl_uids)} | 唯一检查号数 = {len(tbl_accs)}")
PY
```

**判读**

| 看到 | 结论 | 动作 |
|---|---|---|
| `读到的条数` 是几千，取值分布里大量 `T1CE/T2/FLAIR`，`其他` 只占少数 | ✅ 表正常，`其他` 是真的（少数序列） | 那些例靠体素兜底 → 跑 **§2** |
| **取值分布几乎全是 `其他`** | 表把绝大多数序列标成 `其他` | 跑 **§2** 确认体素兜底可用；若一致率低，得看官方表口径 |
| 取值分布是**一堆不像模态的值**（UID、数字、空、`SeriesUid` 字面量） | ❌ **读错列了** | 把本段输出贴出来，需要修 `_read_rows` 的列识别 |
| **`前20例共 0 个序列`（分母为 0）** | ❌ **假信号**：文件不在 `<检查号>/*.nii.gz` 的**一层**结构里 | **不要据此下结论**，跑 **§1b** 看真实布局 |
| `命中 UID = 0%` 但分母 > 0 | 文件名确实不是 `SeriesUid` | 对比 §1b 打印的 `repr(key)` 与真实文件名 → 改成正确的匹配形式 |
| `读到的条数 = 0` | 表没读到或表结构不认 | 检查 §表头原文；或 `export GLIOMA_LABELS_DIR=<表所在目录>` |
| 取值分布里 `其他` 只占几个百分点 | ✅ **表是健康的**，`其他` 是真的（少数异常序列） | 那些例靠体素兜底 → 跑 **§2** |
| 取值分布里出现**两种序列号格式**（纯 `2.25.x` 与 `md5*…*2.25.x*` 混用） | ⚠️ 匹配规则必须同时兼容这两种 | 见 §1b 的 `repr(key)` 输出 |

---

## 1b. 诊断①续 · 目录真实布局（约 10 秒，**纯观测**）

> 只在 §1 出现**分母为 0**（`共 0 个序列`）时跑。它不判断对错，只把事实摊开：
> 文件后缀分布、检查号内第 1 层子目录名、完整树、以及表 key 的 `repr` 原文。

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 - <<'PY'
from pathlib import Path
from collections import Counter
from data.modality_fallback import _read_rows, _table_in

ROOT = Path('/2026aicompetition/datasets/verification')
SUB = ROOT / 'original' if (ROOT / 'original').is_dir() else ROOT
print("ROOT =", ROOT, "| 存在:", ROOT.is_dir())
print("SUB  =", SUB, "| 存在:", SUB.is_dir())
accs = sorted(p for p in SUB.iterdir() if p.is_dir())
print("检查号目录数:", len(accs), "| 样例:", [a.name for a in accs[:2]])

ext, depth1 = Counter(), Counter()
for a in accs[:20]:
    for p in a.rglob('*'):
        if p.is_file():
            ext[''.join(p.suffixes[-2:]) or '<无后缀>'] += 1
        elif len(p.relative_to(a).parts) == 1:
            depth1[p.name] += 1

print("\n--- 文件后缀 top15（下钻全部层级，20 个检查号）---")
if not ext:
    print("  !! 一个文件都没有 → 数据没挂在这个路径，或全在更深层")
for k, v in ext.most_common(15):
    print(f"  {k:<24} {v}")

print("\n--- 检查号目录内的第 1 层子目录名（top10）---")
for k, v in depth1.most_common(10):
    print(f"  {k[:88]:<90} {v}")

print("\n--- 完整树（前 2 个检查号，最多 30 项）---")
for a in accs[:2]:
    print(f"\n[{a.name}]")
    for n, p in enumerate(sorted(a.rglob('*'))):
        rel = p.relative_to(a)
        if len(rel.parts) > 3:
            continue
        tag = "DIR" if p.is_dir() else f"{p.stat().st_size:>11,}B"
        print(f"  {tag}  {rel}")
        if n >= 30:
            print("  …")
            break

table = _table_in(SUB)
rows = _read_rows(table)
print("\n--- 表的 key 原样（repr，看清分隔符与格式）---")
for k in list(rows)[:6]:
    print("  ", repr(k), "->", repr(rows[k]))
PY
```

**怎么读**

| 看到 | 含义 |
|---|---|
| 后缀是 `.nii.gz`，但都在 `<检查号>/<子目录>/` 里 | 代码的文件发现路径少下钻了一层 → 修发现逻辑 |
| 后缀是 `.nii` / `.npz` / `.dcm` / 无后缀 | 后缀假设错了 |
| 后缀分布是**空的** | 数据没挂在 `SUB`；把 §1b 第 2~3 行（`ROOT`/`SUB`）贴出来 |
| `repr(key)` = `('检查号', '2.25.x')` | 匹配规则按**纯 UID** 比 |
| `repr(key)` = `('检查号', 'md5*md5*2.25.x*')` | 匹配规则必须按**复合串**比，且要兼容另一种格式 |

---

## 1c. 诊断①终 · UID 键到底为什么匹配不到（约 15 秒）

> 只在「日志说**序列表找到了**、但按 `(检查号,序列号)` / UID 都匹配不到」时跑。
>
> **前置事实（已核对代码）**：三层布局 `<检查号>/<序列号>/<序列号>.nii.gz` 是被支持的 ——
> `data/loader.py:123-151` 会把 `parts>2` 的按父目录分组、**单文件直接选中**；
> `data/loader.py:492-495` 取 `series_uid = path.parent.name`。
> 所以文件发现与 `series_uid` 取值都没问题，**只剩"拿 UID 去表里查"这一环**。
>
> **关键**：全部用 `repr()` 打印 —— 这样即使复制粘贴把字符拼坏，也能看出哪一段被重复了。

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 - <<'PY' 2>&1 | tee /tmp/modality_probe.txt
import re
from pathlib import Path

from data.loader import DatasetLoader, _nifti_stem
from data.modality_fallback import _read_rows, _table_in, _uid_candidates, _norm
from data.series_selector import guess_modality

ROOT = Path('/2026aicompetition/datasets/verification')
if (ROOT / 'original').is_dir():
    ROOT = ROOT / 'original'
print("ROOT =", ROOT, "| 存在:", ROOT.is_dir())
print("ROOT 子目录:", sorted(p.name for p in ROOT.iterdir() if p.is_dir())[:6])

studies = []
try:
    for st in DatasetLoader().iter_studies(ROOT):
        studies.append(st)
        if len(studies) >= 2:
            break
except Exception as exc:
    print("!! iter_studies 失败:", type(exc).__name__, exc)
print("取到检查号:", [s.accession_number for s in studies])
if not studies:
    raise SystemExit("没有检查号 → 先看上面 ROOT 对不对")

print("\n" + "=" * 78)
print("①  磁盘侧：loader 实际给出的 series_uid / 候选 / sidecar")
print("=" * 78)
for st in studies:
    print(f"\n[{st.accession_number!r}]  序列数={len(st.series)}")
    for s in st.series[:8]:
        print(f"  uid            = {s.series_uid!r}")
        print(f"  modality(描述) = {s.modality!r}   guess={guess_modality(s.modality)!r}")
        print(f"  sidecar UID    = {s.metadata.get('SeriesInstanceUID')!r}")
        print(f"  parent.name    = {s.source_path.parent.name!r}")
        print(f"  stem           = {_nifti_stem(s.source_path)!r}")
        print(f"  candidates     = {list(_uid_candidates(s))}")
        print(f"  source         = {s.source_path}")

print("\n" + "=" * 78)
print("②  表的键（原样 repr）")
print("=" * 78)
table = _table_in(ROOT)
print("table =", table)
if table is None:
    print("!! 表没定位到 → 兜底模块拿不到任何数据")
else:
    rows = _read_rows(table)
    index = {}
    for (_a, u), v in rows.items():
        index.setdefault(u, v)
    print("条数 =", len(rows), "| UID 单键索引 =", len(index))
    st = studies[0]
    acc = _norm(st.accession_number)
    same = {k: v for k, v in rows.items() if k[0] == acc}
    print(f"\n表里属于 [{st.accession_number!r}] 的行：{len(same)} 条")
    for (a, u), v in list(same.items())[:10]:
        print(f"  key_acc = {a!r}")
        print(f"  key_uid = {u!r}   ->  {v!r}")

    print("\n" + "=" * 78)
    print("③  逐候选命中判定 + 最相近的表 uid（差异点就在这里）")
    print("=" * 78)
    pool = [u for (_a, u) in same] or list(index)
    for s in st.series[:8]:
        print(f"\n uid = {s.series_uid!r}")
        for c in _uid_candidates(s):
            n = _norm(c)
            print(f"   候选 {n!r}")
            print(f"     精确键 (acc, 候选) 命中 = {(acc, n) in rows}")
            print(f"     UID 单键           命中 = {n in index}   -> {index.get(n)!r}")
            if n not in index:
                import difflib
                for near in difflib.get_close_matches(n, pool, n=1, cutoff=0.3):
                    print(f"     最相近表 uid = {near!r}")
                    print(f"       磁盘片段 = {[p for p in re.split(r'[^0-9A-Za-z]+', n) if p]}")
                    print(f"       表片段   = {[p for p in re.split(r'[^0-9A-Za-z]+', near) if p]}")
PY
```

**判读（③ 的「磁盘片段 / 表片段」对照）**

| 看到 | 结论 | 修法 |
|---|---|---|
| 表片段**多一段**（如同一 hash 出现两次） | 表 uid 是 `<hash>*<hash>*<UID>*` 复合串 | 加"片段集合匹配"（见下） |
| 磁盘/表**只有尾随 `*` 不同** | 前后缀未归一 | `_norm` 里剥掉首尾非字母数字字符 |
| 片段里**都有同一个 `2.25.x`** | 纯 DICOM UID 是共同锚点 | 按 `2.25.x` 段匹配 |
| `UID 单键索引 << 条数` | 有跨检查号重名的 UID | 精确键优先，别只靠单键 |
| `table = None` | 表没定位到 | 给 `data/loader._series_type_table_path` 补 `GLIOMA_LABELS_DIR` |

**预判的补丁（待 ③ 确认后再落）** —— 给 `modality_fallback._match` 加第三级「片段集合」匹配：

```python
def _uid_segments(value: str) -> frozenset[str]:
    """UID 按非字母数字切段；含 '.' 的 DICOM UID 段优先，用于复合串容错匹配。"""
    parts = [p for p in re.split(r"[^0-9A-Za-z.]+", _norm(value)) if p]
    return frozenset(p for p in parts if "." in p) or frozenset(parts)
```

> ⚠️ 这一级**必须放在最后**（精确键 → UID 单键 → 片段集合），否则会引入误命中：
> 片段匹配比精确匹配宽，只有前两级全落空才用它兜底。

---

## 2. 诊断② · 体素判别与官方表的一致率（约 40 秒，最多 40 例）

**回答**：第三条腿能不能用？`其他` 到底该不该猜？

> 原理：训练集里**表能匹配上**的序列自带权威模态标签 → 拿它当金标准，测体素判别的准确率。
> 体素判别只读 `.nii.gz`，**不需要任何 xlsx**。

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 - <<'PY'
import itertools
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import nibabel
from data.modality_fallback import _read_rows, _table_in
from data.voxel_modality import load_model, describe

SUB = Path('/2026aicompetition/datasets/training/annotation')
table = _table_in(SUB)
print("训练集表:", table, "|", describe())
if table is None:
    raise SystemExit("✗ 训练集里没找到 SeriesType.xlsx")
rows = _read_rows(table)
print("表条数:", len(rows))

model = load_model()
hits = tot = 0
for (acc, uid), label in itertools.islice(rows.items(), 0, 800):
    cands = list(SUB.glob(f"*/{uid}.nii.gz")) or list(SUB.glob(f"*/{uid}*.nii.gz"))
    if not cands:
        continue
    try:
        vol = np.squeeze(np.asanyarray(nibabel.load(cands[0]).dataobj)).astype(np.float32)
    except Exception as exc:                                       # noqa: BLE001
        print(f"  [skip] {uid[:20]}: {type(exc).__name__}")
        continue
    lab, p = model.predict(vol)
    hit = lab.lower().replace('ce', 'c') in str(label).lower().replace('ce', 'c')
    tot += 1
    hits += hit
    if tot <= 12:
        print(f"  {uid[:24]:<26} 表={str(label):<14} 判别={lab:<6} "
              f"{max(p.values()):.2f}  {'OK' if hit else 'X'}")
    if tot >= 40:
        break
print(f"\n与官方表一致: {hits}/{tot} = {hits/max(1,tot):.1%}")
print("判据：>=80% 可用；50~80% 建议重训；<50% 需排查")
PY
```

**一致率低时怎么重训**（**纯 CPU，约 3~6 分钟，不碰 GPU、不碰 `best.pth`**）：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
python3 scripts/31_train_modality_model.py --root /2026aicompetition/datasets/training/annotation
cp -v data/modality_model.json /2026aicompetition/workspace/dcs/Glioma_recognition-main/data/
export GLIOMA_MODALITY_MODEL=/2026aicompetition/workspace/dcs/Glioma_recognition-main/data/modality_model.json
```

> ⚠️ 这只是那个 **~2 KB 的模态判别器**（目标④的 14 个字段头 / 分割骨干完全不受影响）。
> 仓库自带的系数是**本地模拟集**训练的，与官方数据有域差，所以官方文档本来就建议用训练集覆盖重训。

---

## 3. 诊断③ · Goal5 端到端结果（跑完推理后，约 5 秒）

**回答**：通道到底填上了没？掩膜指向的是真实目录吗？

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
OUT=/2026aicompetition/workspace/answer/local-001

echo "=== ① 模态相关日志（体素判别 / 表回贴 / 缺通道）==="
grep -E "\[voxel-modality\]|模态回退|缺通道|goal5" logs/local_eval.log | tail -30

echo
echo "=== ② 答案例数 ==="
ls "$OUT" 2>/dev/null | wc -l

echo
echo "=== ③ 抽查掩膜 URI 是否指向真实文件 ==="
python3 - <<'PY'
import glob
import json
import os

OUT = '/2026aicompetition/workspace/answer/local-001'
ps = sorted(glob.glob(os.path.join(OUT, '*', 'prediction.json')))
print("答案例数:", len(ps))
for p in ps[:3]:
    d = json.load(open(p, encoding='utf-8'))
    base = os.path.dirname(p)
    print(f"\n{p}")
    print("  AccessionNumber:", d.get("AccessionNumber"))
    uri = d.get("SegmentationMaskURI") or {}
    print("  SegmentationMaskURI:", uri)
    for k, u in uri.items():
        f = os.path.normpath(os.path.join(base, u))
        ok = os.path.isfile(f)
        size = os.path.getsize(f) if ok else 0
        print(f"    {k:<6} -> {f}  存在={ok}  {size} B")
        if ok:
            try:
                import nibabel as nib
                img = nib.load(f)
                arr = img.get_fdata()
                print(f"           shape={img.shape} 体素和={int(arr.sum())} "
                      f"（0 表示空掩膜）")
            except Exception as exc:                               # noqa: BLE001
                print(f"           读掩膜失败: {type(exc).__name__}: {exc}")
PY
```

**判读**

| 看到 | 结论 |
|---|---|
| 日志有 `[voxel-modality] study '...' 体素判别 → {...}` | ✅ 第三条腿在生效 |
| 日志有 `goal5: 缺通道 ['flair','t1']（已零占位）` | ⚠️ 部分通道缺 → 该例分割会偏弱，但**不会 0 分作废** |
| `SegmentationMaskURI` 两个键都在，且 `存在=True` | ✅ 输出合规 |
| 掩膜 `体素和=0` | ⚠️ 预测为空（通常是通道全零所致）→ 回到 §1/§2 |
| 答案是**0 例** 或日志有 traceback | 批次被中断 → 看 traceback 最后一层的 `AccessorNumber`，该例单独查 |

---

## 4. 三份诊断怎么组合成结论

| ① 表诊断 | ② 一致率 | ③ Goal5 | 结论 | 动作 |
|---|---|---|---|---|
| 取值分布正常（少量 `其他`） | — | 有 `t1c`/`flair` 填充 | ✅ 上游健康 | 直接往下走（收尾 → 提交） |
| 少数 `其他` | ≥80% | 通道大部分填上 | ✅ 兜底生效 | 可直接用；有空再重训提精度 |
| 少数 `其他` | <80% | 通道部分填上 | ⚠️ 兜底不可信 | 重训模态模型（§2 末） |
| **几乎全是 `其他`** | ≥80% | 通道填上了 | ✅ 表口径如此，兜底是唯一出路 | 用兜底；**并向组委会确认表口径** |
| **几乎全是 `其他`** | <80% | 通道空 | ❌ 上游未解决 | 先重训；仍不行则把 ① 的表头原文贴出来查列识别 |
| 取值**不像模态** | — | — | ❌ **读错列** | 修 `_read_rows` 的列识别（贴 ① 输出） |
| `读到的条数 = 0` | — | — | ❌ 表没读到 | `export GLIOMA_LABELS_DIR=<表所在目录>` 后重试 |
| **`前20例共 0 个序列`（分母 = 0）** | — | — | ❓ **假信号**，暂不可判定 | 跑 **§1b**，按真实布局定位文件发现逻辑 |

---

## 5. 一条命令抓全部现场（排错时直接贴我）

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
{
  echo "===== 时间 ====="; date
  echo "===== 代码版本 ====="; git --no-pager log --oneline -3 2>/dev/null
  echo "===== 关键修复是否在位 ====="
  grep -c "_reference_series"         tasks/goal5_segmentation/task.py
  grep -c "allow_excluded"            data/voxel_modality.py
  grep -c "_any_series_ref"           tasks/goal5_segmentation/preprocess.py
  echo "===== 数据目录 ====="
  ls -d /2026aicompetition/datasets/verification/* 2>/dev/null
  echo "===== 答案产物 ====="
  ls /2026aicompetition/workspace/answer/ 2>/dev/null
  echo "===== 日志尾部 ====="
  tail -40 logs/local_eval.log 2>/dev/null
} | tee /tmp/modality_dump.txt
echo "已写入 /tmp/modality_dump.txt"
```

---

## 6. 一句话

**先跑 §1**（10 秒）——它会直接区分「表值是 `其他`」和「读错列」这两种完全不同的成因；<br>
若 §1 打印 **`共 0 个序列`**（分母为 0）→ 那是**假信号**，接着跑 **§1b** 看真实目录布局；<br>
若日志说**表已找到、但 UID 匹配不到**（本仓真实布局 `<检查号>/<序列号>/<序列号>.nii.gz` 已确认支持）
→ 直接跑 **§1c**，它的「磁盘片段 / 表片段」对照会指出分隔符或前后缀差异；<br>
**再跑 §2**（40 秒）——它决定体素兜底能不能信（≥80% 可用）；<br>
最后跑完推理看 **§3**，用 `missing_channels` 与掩膜的 `体素和` 判断是否真的救回来了。<br>
三份结论按 **§4** 的表组合，就知道下一步该改什么。

> **别把「0%」当结论。** `0/0` 表示"我没找到文件"，不表示"匹配不上"——
> 这两种成因的修法完全不同（前者改文件发现路径，后者改 UID 匹配规则）。
