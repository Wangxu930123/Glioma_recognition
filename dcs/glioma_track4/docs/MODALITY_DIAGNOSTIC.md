# 模态识别 1 分钟诊断（上游问题定位）

> 目的：一条命令定位"**为什么挑不出序列**"，并判断体素兜底能不能用。
> 三个诊断合计约 1 分钟（② 较慢，最多 60 例）。
>
> **路径说明**：文中 `cd` 到**提交侧工程**（Python 包 `data/`、`tasks/`、`pipeline/` 所在的那一层）。
> 若你的工程根不同（例如还在旧位置 `…/dcs/Glioma_recognition-main`，
> 或已迁回 `…/dcs/glioma_track4` 隔壁），把 `cd` 那行换成你自己的路径即可 —— 其余命令不用动。
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
| `[runner] #N … 开始` 后很长时间没有 `完成` | 该例正在跑滑窗推理，**或**真的卡住 | **§3b** |
| `[runner] #N … 完成 Nms goal5{missing=[...], core=…, flair=…}` | 逐例进度（新）：直接看缺通道与掩膜体素数 | **§3b** |
| `[voxel-modality] …（置信 [1.0]，次优 [0.0]，allow_excluded=True，…）` | **概率饱和** + 覆盖了表的 `其他` | **§2** |

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

## 1d. 诊断①全量 · 表行 ↔ 磁盘逐条核对（约 15 秒，**不读影像**）

> **什么时候跑**：§1c 已证明"匹配链路通"（候选命中 `True`），但某个模态的通道仍然是空的。
> 这时要分清两种**完全不同**的成因 —— 它们的修法相反：
>
> | 情况 | 含义 | 处置 |
> |---|---|---|
> | 该模态的表行**能**对上磁盘 | 数据**本来**就缺这个模态 | 数据特性，**不要改代码**；模型必须容忍缺模态 |
> | 该模态的表行**对不上**磁盘 | **匹配 bug**，该通道被人为清零 | 必须修 `_match`（加片段匹配，见 §1c 末尾） |
>
> **只比路径、不读 NIfTI**，所以 1700+ 条几秒跑完。

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 - <<'PY' 2>&1 | tee /tmp/modality_sweep.txt
from collections import Counter, defaultdict
from pathlib import Path

from data.modality_fallback import _read_rows, _table_in, _norm
from data.series_selector import guess_modality

ROOT = Path('/2026aicompetition/datasets/verification')
if (ROOT / 'original').is_dir():
    ROOT = ROOT / 'original'
table = _table_in(ROOT)
rows = _read_rows(table)
print("表 =", table, "| 条数 =", len(rows))

by_dir, by_stem, n_dir = {}, {}, 0
for acc_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for uid_dir in sorted(p for p in acc_dir.iterdir() if p.is_dir()):
        n_dir += 1
        by_dir.setdefault((_norm(acc_dir.name), _norm(uid_dir.name)), uid_dir)
        for f in uid_dir.glob('*.nii*'):
            stem = f.name[:-7] if f.name.lower().endswith('.nii.gz') else f.stem
            by_stem.setdefault((_norm(acc_dir.name), _norm(stem)), f)
print(f"磁盘：检查号 {len({a for a, _ in by_dir})} 个 | 序列目录 {n_dir} 个 | 文件名主干 {len(by_stem)} 条")

stat = defaultdict(lambda: [0, 0, 0])
miss = []
for (acc, uid), label in rows.items():
    ok_dir = (acc, uid) in by_dir
    ok_stem = (acc, uid) in by_stem
    s = stat[label]
    s[2] += 1
    s[0] += ok_dir
    s[1] += ok_stem
    if not (ok_dir or ok_stem) and len(miss) < 12:
        miss.append((acc, uid, label))

print("\n--- 按模态统计：表行 → 磁盘是否真有该序列 ---")
print(f"  {'模态':<16} {'表行':>6} {'目录名命中':>16} {'文件名命中':>16} {'模态键':>8}")
for label, (d, st, n) in sorted(stat.items(), key=lambda kv: -kv[1][2]):
    print(f"  {label:<16} {n:>6} {d:>7} ({d / n:6.1%}) {st:>7} ({st / n:6.1%}) {str(guess_modality(label)):>8}")

print("\n--- 未命中样例（前 12）---")
for acc, uid, label in miss:
    print(f"  acc={acc!r}  uid={uid!r}  label={label!r}")
n_miss = sum(1 for (a, u) in rows if (a, u) not in by_dir and (a, u) not in by_stem)
print("未命中总数 =", n_miss, "/", len(rows))

# 找一个表里**有 T1CE** 的检查号，把表行与磁盘目录并列打印
print("\n--- 抽样：表里有 T1CE 的检查号，表行 vs 磁盘目录名 ---")
targets = [a for (a, u), l in rows.items() if 't1c' == guess_modality(l)]
if targets:
    acc = targets[0]
    print(f"\n[{acc}]  表行：")
    for (a, u), l in rows.items():
        if a == acc:
            print(f"   {u!r}  ->  {l!r}")
    print("  磁盘目录名：")
    for uid_dir in sorted(p for p in (ROOT / acc).iterdir() if p.is_dir()) if (ROOT / acc).is_dir() else []:
        files = [f.name for f in uid_dir.glob('*.nii*')]
        print(f"   {uid_dir.name!r}   文件={files}")
    print(f"  → 表行的 uid 是否能在磁盘目录名里找到："
          f"{[(_norm(u) in {_norm(d.name) for d in (ROOT / acc).iterdir() if d.is_dir()}) for (a, u) in rows if a == acc]}")
else:
    print("!! 表里没有一条 T1CE → 结论更直接：数据/表本身就缺 t1c")
PY
```

**判读**

| 看到 | 结论 | 动作 |
|---|---|---|
| 某模态命中率 **≈100%** | 数据**真的**缺这个模态 | 不改代码；报告里说明"缺模态条件下的指标" |
| 某模态命中率 **明显 <100%** | **匹配 bug**，该通道被人为清零 | 修 `_match`（§1c 末尾的片段匹配） |
| 全表命中率都低 | 目录结构 / `_norm` 口径问题 | 把输出贴出来 |
| 表里**一条某模态都没有** | 数据/表本身缺该模态 | 同上第一行 |

---

## 1e. 已验证：表取值 → 模态键的映射（无需再查）

对本数据集表的 **5 个取值**逐个实测（`data/series_selector.guess_modality`）：

| 表取值 | 归一结果 | 结论 |
|---|---|---|
| `T2-Flair` | `flair` | ✅ |
| `T1CE (增强)` | `t1c` | ✅ |
| `T1` | `t1` | ✅ |
| `T2WI` | `t2` | ✅ |
| `其他` | `None` | ✅ 正确排除 |

变体也全对：`T1CE（增强）` / `t1ce(增强)` / `T1 CE` / `CE增强` → `t1c`；
` T2-FLAIR ` / `T2Flair` / `FLAIR` → `flair`；`T1WI` → `t1`。
`_key_of` 分支也正确：`metadata['modality']='其他'` → `None`（不短路后续兜底）。

> **所以：`表 → 描述 → 模态键` 这条链是通的。若某通道仍为空，病因只可能在 §1d 的"表行对不上磁盘"。**

### ⚠️ 但 `t1c` 的关键词里有**过宽的词**（已知隐患，本数据集暂不触发）

```21:22:Glioma_recognition-main/data/series_selector.py
    ("t1c", ("t1c", "t1ce", "t1_ce", "t1+c", "t1wi+c", "postcontrast", "post_contrast",
             "post contrast", "post", "enhance", "增强", "ce+", "+c", "gd")),
```

`"post"` / `"gd"` / `"+c"` 都是**子串**匹配，且 `t1c` 排在关键词表**第一位**（顺序敏感设计），
一旦命中就短路，后面的 `flair/t2/t1` 全被吃掉。实测：

```
'post contrast t2' -> 't1c'   ← 错，应为 t2
'gd-t2'            -> 't1c'   ← 错
'T2WI+c'           -> 't1c'   ← 含糊
```

`"post"` 会吞 `posterior` / `post-op`，`"gd"` 会吞任何含 `gd` 的串。
本数据集的 5 个取值不含这些词，所以**当前不触发**；但官方表/测试集描述一变就会踩。
收窄建议：删掉裸 `"post"`（`postcontrast`/`post contrast` 已单列），`"gd"` 改 `"gd-"`/`"-gd"`/`"gd "`。

---

## 1f. 诊断①尾 · 「挑不出序列」到底卡在哪一阶段（约 10 秒）

### 先分清 4 种来源 —— **只有 3 种是真失败，1 种是预期行为**

| 日志原文 | 出处 | 性质 |
|---|---|---|
| `[selector][模态回退] 挑不出序列，且未找到数据集自带的 SeriesType.xlsx…` | `modality_fallback` 的 `warned_missing` | ❌ **真失败**：表没定位到 |
| `[selector][模态回退] 挑不出序列，序列表在 X 但按 (检查号,序列号) / UID 都匹配不到` | `modality_fallback` 的 `warned_unmatched` | ❌ **真失败**：UID 对不上 |
| `goal5: 挑不出 t1c/t1 模态 → 掩膜退化写入参考序列 …（答案仍合规但该例分割可能不准）` | `goal5_segmentation/task.py:152-156` | ⚠️ **预期行为**：缺 T1 增强的检查（约 47%）**必然**出现这一行 |
| `ValueError: study … 没有任何可用影像（共 N 条…）` | `goal5_segmentation/preprocess.py:90` | ❌ **真失败**：一条影像都没读进来 |

> **判据**：前两条说「**序列**」，第三条说「**模态**」（t1c/t1 这种）。
> 看到第三条不要慌 —— 那是数据缺模态 + 退化写掩膜正常工作的证据，不是 bug。

### 分阶段定位脚本

`select()` 有**四条腿**依次兜底，`_select_picked` 返回空就是该阶段失败。
这段把每一阶段的输出都打出来，**第一个空的阶段就是失败点**：

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main
python3 - <<'PY' 2>&1 | tee /tmp/series_pick_trace.txt
from pathlib import Path

from data.loader import DatasetLoader
from data.modality_fallback import (
    _table_maps, _uid_candidates, _norm, find_label_table, recover_study)
from data.series_selector import _key_of, _select_picked, select
from data.voxel_modality import recover_study as recover_by_voxels
from tasks.goal5_segmentation.preprocess import CHANNEL_ORDER

ROOT = Path('/2026aicompetition/datasets/verification')
if (ROOT / 'original').is_dir():
    ROOT = ROOT / 'original'
print("ROOT =", ROOT, "| CHANNEL_ORDER =", CHANNEL_ORDER)

studies = []
for i, st in enumerate(DatasetLoader().iter_studies(ROOT)):
    studies.append(st)                      # 只取前 3 例，内存可控
    if i >= 2:
        break

table = find_label_table([s.source_path for st in studies for s in st.series])
rows, index = _table_maps(table)
print("表定位 =", table)
print(f"表条目 = {len(rows)} | UID 单键索引 = {len(index)}")

for st in studies:
    print("\n" + "=" * 76)
    print(f"[{st.accession_number}]  序列数={len(st.series)}")
    for s in st.series:
        print(f"  desc = {s.modality!r}")
        print(f"    _key_of  -> {_key_of(s)!r}")
        print(f"    sidecar  -> {(s.metadata or {}).get('SeriesInstanceUID')!r}")
        print(f"    候选     -> {list(_uid_candidates(s))}")

    print("--- 分阶段 select：第一个空的就是失败点 ---")
    print(f"  stage0 原始描述       -> {sorted(_select_picked(st, CHANNEL_ORDER))}")
    print(f"  stage1 官方表重贴     -> {sorted(_select_picked(recover_study(st), CHANNEL_ORDER))}")
    print(f"  stage2 体素判别       -> {sorted(_select_picked(recover_by_voxels(st), CHANNEL_ORDER))}")
    print(f"  stage3 连「其他」也猜 -> "
          f"{sorted(_select_picked(recover_by_voxels(st, allow_excluded=True), CHANNEL_ORDER))}")
    print(f"  select() 最终         -> {sorted(select(st, CHANNEL_ORDER))}")

    acc = _norm(st.accession_number)
    same = {k: v for k, v in rows.items() if k[0] == acc}
    print(f"--- 表里属于该检查号的行：{len(same)} 条 ---")
    for (a, u), v in list(same.items())[:8]:
        print(f"  表 uid len={len(u):>3} -> {v!r}")

    if st.series:
        print("--- 逐候选命中判定（用长度，绕开 Markdown 吃掉 * 的问题）---")
        for s in st.series[:3]:
            for c in _uid_candidates(s):
                n = _norm(c)
                print(f"  候选 len={len(n):>3}  精确键={(acc, n) in rows}  单键={n in index}")
PY
```

**判读**

| 现象 | 成因 | 动作 |
|---|---|---|
| `表定位 = None` | 表没找到 | 设 `GLIOMA_SERIES_TYPE_XLSX=<表路径>` 或 `GLIOMA_LABELS_DIR=<表所在目录>` |
| `表条目 = 0` | 表读不出来 | 跑 §1 看表路径与表头原文 |
| 所有 `候选 ... 精确键=False 单键=False` | 表与磁盘的 UID 口径不一致 | 贴输出（含 `len`）→ 需要第三级「片段匹配」 |
| **stage0 空、stage1 非空** | 原描述认不出，靠官方表救回 | ✅ 正常（正是兜底在工作） |
| **stage1 也空** | 表匹配不上 | 同上第 3 行 |
| **stage1 有 t1c/flair，但 stage3 仍缺 t1c/t1** | 该检查**真的**没有这些模态 | ⚠️ 预期行为（约 47%），不是 bug |
| 全阶段都空 | 序列被标 `其他` 或只有 DWI/ADC/SWI | 看 `goal5: 缺通道 [...]` 告警；属数据特性 |

### 已知风险（**已修**）：UID 候选集塌缩

`data.loader` 按官方契约把 `Series.series_uid` 定为**磁盘目录名**之后，
`_uid_candidates` 里 `series_uid` / `parent.name` / `stem` **三者变成同一个值** ——
去重后只剩 1 个候选，**sidecar 的 `SeriesInstanceUID` 再也进不了候选集**。

而官方表的 `SeriesUid` 列写的是哪一边，数据方**并没有承诺**（实测带装饰的 `*2.25.…*`
与纯 `2.25.…` 两种形态都存在）。所以已在 `_uid_candidates` 里**把 sidecar 值加回候选**
（只排序、不裁剪）—— 否则本来能匹配上的检查会直接变成「挑不出序列」。

---

## 2. 诊断② · 体素判别的**准确率 + 概率饱和度**

> 脚本：`scripts/voxel_consistency.py`（**不占 GPU、不读权重**，可与推理并行）
> 前台约 40 秒（默认 60 例）；后台用 nohup 见 §2.1

**回答两个问题**：第三条腿能不能用？`其他` 到底该不该猜？

> 原理：训练集里**表能匹配上**的序列自带权威模态标签 → 拿它当金标准，测体素判别的准确率。
> 体素判别只读 `.nii.gz`，**不需要任何 xlsx**，也**不碰 GPU**。
>
> ⚠️ **只看准确率不够** —— 必须同时看 **top-2 间隔（饱和度）**：见下面的判据。

### 2.1 后台跑（推荐：nohup，不占终端）

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main

# ① 先确认脚本在位（它随仓库一起走）
test -f scripts/voxel_consistency.py && echo "脚本就位" \
  || echo "!! 缺少 scripts/voxel_consistency.py —— 先同步代码再跑"

# ② 后台启动，日志落盘
mkdir -p logs
nohup python3 scripts/voxel_consistency.py > logs/voxel_consistency.log 2>&1 &
echo $! > logs/voxel_consistency.pid
echo "已后台启动，PID=$(cat logs/voxel_consistency.pid)"
```

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main

# 实时跟进（Ctrl-C 只退出 tail，**不影响**后台任务）
tail -f logs/voxel_consistency.log

# 只看结论几行
grep -E "一致率|top-2 间隔|按表标签|判据" logs/voxel_consistency.log

# 确认后台任务是否还在跑（无输出 = 已跑完）
ps -o pid,etime,time --no-headers -p "$(cat logs/voxel_consistency.pid)"
```

### 2.2 前台跑 / 调规模

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main

# 前台直接跑（默认 60 例）
python3 scripts/voxel_consistency.py

# 指定数据根 + 最多比对 300 例（例数越多，一致率越可靠）
python3 scripts/voxel_consistency.py /2026aicompetition/datasets/training/annotation 300

# 后台 + 300 例
GLIOMA_VOXEL_MAX=300 nohup python3 scripts/voxel_consistency.py \
  > logs/voxel_consistency.log 2>&1 &
```

| 参数 | 位置 | 默认 |
|---|---|---|
| 数据根 | 第 1 个位置参数 | `/2026aicompetition/datasets/training/annotation` |
| 最多比对例数 | 第 2 个位置参数 / `GLIOMA_VOXEL_MAX` | `60` |
| 外部模型 | `GLIOMA_MODALITY_MODEL` | 不设 → 用**内嵌系数** |

**退出码**：`0` = 正常跑完；`1` = 中途异常（回溯已打在日志末尾）；`2` = 数据根下没有 `SeriesType.xlsx`。

### 2.3 输出长什么样（**格式示意**，数字是编的）

```text
[1/5] 数据根: /2026aicompetition/datasets/training/annotation | 存在=True
      体素模型: 体素判别模型=内嵌系数（默认）
      最多比对: 60 例（第二个参数或 GLIOMA_VOXEL_MAX 可改）
[2/5] 序列表: /2026aicompetition/datasets/training/annotation/SeriesType.xlsx
      表条数: 1735
[3/5] 磁盘影像数: 6980   索引键数: 6980
  SR-A-0001                表=T1CE (增强)      判别=T1CE  top=0.998 次优=0.002 OK
  SR-B-0002                表=T2-Flair       判别=FLAIR top=0.991 次优=0.009 OK
      [miss] acc='acc002' uid='ghost'
[4/5] 参与比对 60 例；表里有但磁盘找不到文件而跳过 3 行
      一致率: 48/60 = 80.0%
      top-2 间隔：中位数=0.976 最小=0.213 饱和(<0.01)占比=0.0%
      按表标签分组（看是哪一类在拖后腿）：
        T1CE (增强)        20/24 = 83.3%
        T2-Flair          28/36 = 77.8%
[5/5] 判据：一致率 >=80% 且 饱和占比 <20% → 可覆盖「其他」；一致率 <50%（≈随机）→ 不该覆盖
```

> **重点看每行的 `top` 与 `次优`**。若你看到的输出是
> `（置信 [1.0], 共 N 条序列）` 而**没有 `次优`**，说明跑的是**旧代码**（见 §3c）。

### 2.4 参数与脚本位置

脚本：`scripts/voxel_consistency.py`（**不占 GPU、不读权重**，可与推理并行跑）。
它的逻辑与本文件 §2 完全同源 —— 早期版本是内联 heredoc，现已收进脚本，
**以后只维护脚本一处**。

### 2.5 ⚠️ 先确认「用的是哪个模型」—— 否则数字没有意义

`data/voxel_modality.load_model()` 的取值顺序：**`GLIOMA_MODALITY_MODEL` → 内嵌系数**。
两者差距极大，**实测对照**：

| 模型来源 | 训练数据 | 一致率 |
|---|---|---|
| **内嵌系数**（`voxel_modality._MEAN/_STD/_WEIGHTS/_BIAS`） | 本地**模拟集** | **≈27%**（**低于三分类随机 33%**） |
| **重训后外部模型**（`glioma_track4/data/modality_model.json`） | 官方训练集 | **5 折 CV = 0.735** ✅ |

> **"重训完还是 27%" 的典型原因是 `GLIOMA_MODALITY_MODEL` 没导出 —— 测的仍是内嵌系数。**

脚本第 2 行就会告诉你用的是哪个：

```text
      体素模型: 体素判别模型=内嵌系数（默认）                    ← 27% 的那次多半是这一行
      体素模型: 体素判别模型=/xxx/modality_model.json（外部）
```

`start.sh` 现已**自动挂载** `../glioma_track4/data/modality_model.json`，缺失时会打 `!!` 报警。

**正确的对照测法**（同一批数据、同一例数，才可比）：

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main
ROOT=/2026aicompetition/datasets/training/annotation
MODEL=/2026aicompetition/workspace/dcs/glioma_track4/data/modality_model.json

# A) 外部（重训后）—— 期望接近 0.735
GLIOMA_MODALITY_MODEL="$MODEL" \
  python3 scripts/voxel_consistency.py "$ROOT" 200 2>&1 | tee logs/voxel_acc_external.log

# B) 内嵌（对照）—— 不设 GLIOMA_MODALITY_MODEL 就是它
python3 scripts/voxel_consistency.py "$ROOT" 200 2>&1 | tee logs/voxel_acc_embedded.log

grep -E "体素模型|一致率|top-2" logs/voxel_acc_external.log logs/voxel_acc_embedded.log
```

> **预期**：A 应显著高于 B。若 A 仍 ≈ B ≈ 27%，说明**外部模型没被加载**
> （路径写错 / 文件不在）—— 先看 A 日志里「体素模型」那一行。

**对 §3 那个开关的影响（修正此前的建议）**：

先前基于 27%（低于随机）把 `GLIOMA_VOXEL_GUESS_EXCLUDED` 默认关闭是对的；
**但 0.735 说明这个赌注的期望是正的**。所以：

| 运行时实测一致率 | `GLIOMA_VOXEL_GUESS_EXCLUDED` |
|---|---|
| <50% | `0`（当前 `start.sh` 默认，尊重官方表的 `其他`） |
| **>=70% 且确认加载了外部模型** | **改回 `1`**：`GLIOMA_VOXEL_GUESS_EXCLUDED=1 ./start.sh` |

> 判据用**运行时实测**，不是重训脚本的自报值 —— 两者相差 27% vs 73.5% 就是例子。

**判读**

> ### 如果输出停在某一步（这次遇到的"只到 `表条数` 就没了"）
>
> 脚本刻意打了 `[1/5]`~`[5/5]` 五个进度点，**停在哪一步就能直接定位**：
>
> | 停在 | 含义 | 处理 |
> |---|---|---|
> | `[1/5]` 之后 | 数据根不存在（那行会打 `存在=False`） | 改 `SUB` 路径 |
> | **`[2/5]` 之后（只到 `表条数`）** | **正在建索引**：`SUB.iterdir()` + 每个检查号 `rglob` 遍历整棵树 | ⚠️ **是慢，不是死** —— 见下 |
> | `[3/5]` 之后 | 正在逐例读 NIfTI 并判别（每例约 71 MB 读盘） | 正常，60 例约 40 秒 |
> | 出现 `!!!!` 前缀 | 抛异常，**完整回溯就在下一行** | 整段贴回来 |
>
> **「建索引」是唯一可能明显耗时的一步。** 想确认它只是慢，另开终端量一下：

```bash
R=/2026aicompetition/datasets/training/annotation
echo "检查号目录数: $(ls "$R" | wc -l)"
echo "影像文件数  : $(find "$R" -name '*.nii*' | wc -l)"
```

> 几千个文件时「建索引」应**几秒内**出现 `[3/5]`。若超过一分钟仍没有，
> 用 `Ctrl-C` 中断并把已打印的前几行贴回来（那说明该目录下混进了别的大子树）。

| 看到 | 结论 | 动作 |
|---|---|---|
| 一致率 **≥80%**，`饱和(<0.01)占比` **<20%** | 判别器可信 | 保留"覆盖 `其他`"的逻辑 |
| 一致率 **≥80%**，但**饱和占比很高** | "看起来准"但概率不可信 | 见下方「为什么必须看饱和度」 |
| 一致率 **50~80%** | 边缘 | 用训练集重训（见下） |
| 一致率 **<50%**（≈随机猜） | ❗ **置信 1.0 毫无意义** | **应关掉对 `其他` 的覆盖** |

### 为什么必须同时看饱和度

`voxel_modality.py` 用 0.5 置信门槛兜住乱猜，注释写着「把 FLAIR 当 T1C 比留空通道更有害」。
**但对 3 类逻辑回归，概率压到 1.0 说明 top-2 的 logit 差极大 —— 这通常是"输入远超训练分布"
（特征标准化后落在极端区）的特征，而不是"非常确信"。** 实跑日志里已经出现过：

```text
[voxel-modality] study '...' 体素判别 → {'2.25.2588...': 'T1CE'}（置信 [1.0]，allow_excluded=True，共 2 条序列）
```

此时 **0.5 的门槛形同虚设**。而那种场景恰恰是**表明确标了 `其他`**的序列 ——
它们最可能是定位像 / 非脑 / calibration 之类的异常序列，
**把定位像当 T1CE 填进 `t1c` 通道，比留空通道更有害**。

> 所以 `voxel_modality` 的日志已经改成同时打 **`次优`** 与 **`allow_excluded`**：
> `置信 1.0 / 次优 0.0` = **饱和**（不可信）；`置信 0.9 / 次优 0.1` = 正常高置信。
> 看到饱和占比高，就该考虑关掉对 `其他` 的覆盖。

### 若结论是「不该覆盖」

目前**没有**单独开关，覆盖发生在 `data/series_selector.select()` 的**最后一句**：

```python
    # 最后一遍：连"权威排除（表里明写 `其他`/`正常`/`平扫`）"的序列也允许猜。
    return _select_picked(recover_by_voxels(study, allow_excluded=True), wanted)
```

两种处置（都不影响其他任何路径）：

| 处置 | 效果 | 代价 |
|---|---|---|
| 把最后一句改成 `return {}` | 尊重表的「权威排除」，`其他` 序列一律不猜 | **整例全 `其他`** 的检查会变成通道全零（**该例分割 0 分**） |
| 加一个环境开关（约 10 行，可随时回切） | 同上，但保留一条"再打开"的路 | 需要改代码 |

> 判据是**期望值**：设判别器对 `其他` 序列的准确率为 `p`，则
> 覆盖的收益 ≈ `p × 救回的分数`，代价 ≈ `(1−p) × 用错通道造成的额外损失`。
> `p` 就在 §2 的「一致率」里 —— 但注意它是在**表有真标签**的序列上测的，
> 对**异常序列（`其他`）**这个 OOD 子集只会更低。所以一致率 <50% 时应当关掉。

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

## 3b. 「卡住了」还是「很慢」？—— 先分清，别急着 Ctrl-C

**这一节是踩过坑写下来的**：一次 700+ 例的推理要跑很久，而 runner 原来**每例之间不打任何日志** ——
于是「进程死锁」和「某例正在跑分钟级滑窗推理」在日志上**完全一样**，只能靠猜。

### 现在 runner 每例打两行

```text
[runner] #1 04c210eaffa444e9b02ef1aaefda4365 开始 …
[runner] #1 04c210eaffa444e9b02ef1aaefda4365 完成 41230ms goal5{missing=[], core=1234, flair=5678}
```

| 看到 | 结论 | 动作 |
|---|---|---|
| `#N … 开始` 后**没有** `完成`，已过几分钟 | 正在跑该例的滑窗推理（分钟级） | **等**；用下面的命令确认它在吃 CPU/GPU |
| `#N … 开始` 后没有 `完成`，且**进程 CPU 时间不再增长** | ❗ 真的卡住（死锁 / 等 IO） | `Ctrl-C`，贴出最后 20 行 |
| **完全没有** `[runner]` 行 | 卡在 **loader 首帧**（建索引 / 读表 / 扫不到 NIfTI） | 见 §2 的「停在某一步」 |
| `完成` 里有 `goal5{missing=[t1c]}` | 缺通道（数据特性） | 正常，见 §0 与 §1f |
| `完成` 里有 **`core=0` 或 `flair=0`** | ❗ 掩膜为空 | 需要查（通道 / 阈值 / 后处理） |

### 不看日志也能判断的 3 条命令（另开一个终端）

```bash
PID=$(pgrep -f "uvicorn app.server" | head -1)
echo "PID=$PID"
ps -o pid,etime,time,stat,rss --no-headers -p "$PID"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
sleep 10
echo "--- 10 秒后再看一次 ---"
ps -o pid,etime,time,stat,rss --no-headers -p "$PID"
```

**判据**：`TIME` 是累计 CPU 时间 —— **它在增长就说明进程在干活**（哪怕日志一行不出）。
两次 `TIME` 相同且 GPU 利用率 0 → 才可能是真的卡住。

关掉逐例进度（日志太长时）：`GLIOMA_PROGRESS=0 ./start.sh`。

### 「有 `完成` 但没有答案」—— 答案在**隐藏的** staging 目录里

**这不是故障，是路径可见性问题。** `OutputWriter` 用**两段式**：

| 阶段 | 路径 | 可见性 |
|---|---|---|
| **运行期间** | `<answer_root>/.<evaluation_id>.tmp-<uuid>/<检查号>/…` | ❗ **以 `.` 开头 → 普通 `ls` 看不见** |
| 全部跑完后 | `<answer_root>/<evaluation_id>/`（staging 被 `rename` 过来） | 这时才可见 |

所以运行中途 `ls /2026aicompetition/workspace/answer/` **是空的**，
而正式目录要等**全部检查跑完**才出现。

```bash
WS=/2026aicompetition/workspace

# ① 用 -a 看（关键：隐藏目录）
ls -la "$WS/answer/"

# ② 找 staging
ST=$(find "$WS/answer" -maxdepth 1 -type d -name '.*.tmp-*' 2>/dev/null | head -1)
echo "staging = $ST"

# ③ 已完成例数（应与日志里的 #N 对得上）
echo -n "已完成: "; find "$ST" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l

# ④ 抽一例看实际产物
ACC=$(find "$ST" -maxdepth 1 -mindepth 1 -type d | head -1)
echo "--- $ACC ---"; ls -lR "$ACC"; cat "$ACC/prediction.json"
```

预期结构：

```text
<staging>/
├── <检查号>/
│   ├── prediction.json
│   ├── <core_uid>/<core_uid>.nii.gz      ← 核心区掩膜
│   └── <flair_uid>/<flair_uid>.nii.gz    ← 周围区掩膜
├── <检查号>/...
└── duplicate_pairs.jsonl                 ← **只在最后**写
```

**`完成 Nms` 是在 `write_study()` + `validate_study()` 都成功之后才打印的** ——
所以那一例的目录必然已经存在。若确实没有，只有三种情况：

| 现象 | 原因 | 处置 |
|---|---|---|
| `goal5{core=0, flair=0}` | 掩膜**写出来了但是空的** | 查通道缺失 / 阈值 / 后处理 |
| 该检查号目录**完全不存在** | 那行"完成"是**别的检查号**的 | 按检查号逐个对 |
| 进程被 `Ctrl-C` / OOM 杀掉 | staging 被 `rmtree`（**设计如此**：半成品不发布） | 见 §3c 重启；或打开 `GLIOMA_LOADER_TOLERANT=1` 只丢单例 |

**顺带把耗时换算**：`完成 25581ms ≈ 25.6 s/例`，777 例 ≈ **5.5 小时**。
若平台有时限，按性价比依次调：`tta_flips` 减轴 → 滑窗 `overlap` 0.5→0.3 → `tta_batch` 调大。

---

## 3c. 重启 + 确认新代码生效（改过 `data/`、`core/`、`start.sh` 后必做）

**为什么必须重启**（两者都**不能**热加载）：

| 状态 | 位置 | 求值时机 |
|---|---|---|
| `GLIOMA_LOADER_TOLERANT` | `data/loader.py` 的 `_TOLERANT` | **模块导入期** —— 进程起来后再 `export` **不生效** |
| `GLIOMA_PROGRESS` | `core/runner.py` 的 `_PROGRESS` | 同上 |
| `voxel_modality._LOGGED`（日志去重） | 进程内状态 | 新进程才会是空的 |

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main

# ① 看清现有进程（确认没杀错）
pgrep -af "uvicorn app.server"

# ② 停掉。若同机还有别的 uvicorn 服务，就别用 pkill，改用 kill <上面列出的 PID>
pkill -f "uvicorn app.server" && echo "已停止" || echo "没有找到在跑的进程"

# ③ 启动（start.sh 内已 : "${GLIOMA_LOADER_TOLERANT:=1}" 并 export）
mkdir -p logs
nohup ./start.sh > logs/start.log 2>&1 &
echo $! > logs/start.pid

# ④ 确认服务活着
sleep 5
curl -s http://127.0.0.1:8000/health; echo
```

**⑤ 确认新代码真的生效**（直接问进程，不靠肉眼看日志）：

```bash
python3 -c "
from data import voxel_modality as vx
from data.loader import tolerant_mode
print('容错开关 tolerant_mode() =', tolerant_mode(), '(期望 True)')
print('日志去重 _LOGGED 存在   =', hasattr(vx, '_LOGGED'), '(期望 True)')
"
```

**日志里出现这些 = 新代码生效**

| 标志 | 说明 | 出处 |
|---|---|---|
| `[runner] #1 … 开始 …` / `完成 Nms goal5{…}` | 逐例进度生效 | §3b |
| 体素判别行含 **`次优`** 与 **`allow_excluded=`** | 新格式生效 | §2 |
| 同一 study 的体素判别**只出现 1~2 次** | 日志去重生效 | 旧版会重复 **8 次** |
| `warned_unmatched` 告警结尾有 **`【本进程只报这一次…】`** | 新文案生效 | §1f |

> ⚠️ **反过来的判据同样有用**：若体素判别行仍是 `（置信 [1.0], 共 N 条序列）`
> —— 没有 `次优`、没有 `allow_excluded` —— 那说明**跑的还是旧代码**，
> 这时看到的一切重复/噪声都是已修过的老问题，不必再分析。

---

## 3d. 「`core=0` / `flair=0`」三种成因怎么分

`core=0, flair=0` 意味着**该例分割直接 0 分**，但它有三种**修法完全不同**的成因：

| 成因 | 判据 | 修法 |
|---|---|---|
| **① 通道零占位** | `missing` **非空** | 数据本身缺该模态（见 §1f）；模型要能容忍缺模态 |
| **② 模型输出偏低** | `pmax` **低于** `thr` | 权重 / 预处理不匹配（通道顺序、归一化、权重是否真为训练产物） |
| **③ 后处理吃掉了** | `pmax` 够高，**阈值以上 > 0** 而**终值 = 0** | 调 `min_tumor_voxels` / `keep_components` / `bridge_mm` |

### 进度行已带全部判据

`[runner] … 完成 Nms goal5{…}` 现在的格式：

```text
goal5{missing=[], thr=[0.42, 0.55], pmax=[0.99, 0.96], core=1480/1520, flair=3050/3100}
       ↑ 缺通道     ↑ 阈值           ↑ 最大概率         ↑ 终值/阈值以上
```

**三种情况一眼可分**（实测格式）：

```text
① 通道缺 + 概率低   goal5{missing=['t1','t2'], thr=[0.42,0.55], pmax=[0.31,0.08], core=0/0,       flair=0/0}
② 后处理吃掉        goal5{missing=[],          thr=[0.42,0.55], pmax=[0.97,0.93], core=0/12,      flair=0/25}
③ 正常              goal5{missing=[],          thr=[0.42,0.55], pmax=[0.99,0.96], core=1480/1520, flair=3050/3100}
```

### 一个已经能确定的推论

读 `postprocess.clean_mask` 可以确定：**归零只可能发生在「阈值以上体素 < `min_tumor_voxels`(=30)」时**。

```51:57:Glioma_recognition-main/tasks/goal5_segmentation/postprocess.py
    m = np.asarray(prob) > float(threshold)
    if not m.any():
        return m.astype(bool)
    if bridge_mm > 0:
        m = _bridge(m, bridge_mm, spacing)
    m = _largest_components(m, max(1, int(keep_components)))
    return m if int(m.sum()) >= int(min_voxels) else np.zeros_like(m, dtype=bool)
```

- `_bridge` 是**形态学闭运算**（先膨胀后腐蚀），数学上满足 `A ⊆ closing(A)` —— **只会增加、不会清空**；
- `_largest_components` 只会**保留最大的几个**，不会归零；
- 唯一能归零的是**最后那一行**：`阈值以上体素 < min_tumor_voxels(30)`。

> **所以 `core=0` 几乎必然是「模型输出概率低于阈值」（成因 ②），不是后处理。**

### ⚠️ `missing` **不是**空掩膜的原因 —— 先做这个交叉统计

**常见误判**：看到 `goal5{missing=['t1c','t2','t1'], core=0, flair=0}` 就以为"缺模态导致空掩膜"。
但表里的覆盖率本来就低（**T1 ≈32%、T2WI ≈31%、T1CE ≈53%**），
所以 `missing` 里有 `t1`/`t2`/`t1c` 是**常态**。真正的判据是这个**交叉统计**：

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main
python3 - <<'PY'
import collections
import re

cnt, empty = collections.Counter(), collections.Counter()
pat = re.compile(r"goal5\{missing=(\[[^\]]*\]).*?core=(\d+), flair=(\d+)\}")
for line in open("logs/start.log", encoding="utf-8", errors="ignore"):
    m = pat.search(line)
    if not m:
        continue
    key = m.group(1)
    cnt[key] += 1
    if m.group(2) == "0" and m.group(3) == "0":
        empty[key] += 1

print(f"{'missing':<30}{'总数':>6}{'空掩膜':>8}{'空掩膜率':>10}")
for k, n in cnt.most_common():
    print(f"{k:<30}{n:>6}{empty[k]:>8}{empty[k] / n:>10.1%}")
tot, emp = sum(cnt.values()), sum(empty.values())
print(f"\n合计 {tot} 例，空掩膜 {emp} 例 = {emp / max(1, tot):.1%}")
PY
```

| 结果 | 结论 |
|---|---|
| 各 `missing` 组合的空掩膜率**都差不多** | ✅ **`missing` 不是原因** —— 问题在模型输出（成因 ②）→ 跑下面的探针确认 `pmax` |
| 只有"缺 `t1c`"那几组空掩膜率显著更高 | 缺模态是**部分**原因 → 回到 §1f 的模态识别线 |
| 全组都接近 100% | 模型输出普遍偏低 → 查**权重真伪 / 预处理是否与训练一致** |

**实际数据里已出现决定性反例**（10 例样本）：

| # | `missing` | 实际填上的通道 | core | flair |
|---|---|---|---|---|
| 18 | `t1c,t2,t1` | `flair` | **2503** | 2503 ✅ |
| 19 | `t1c,t2,t1` | `flair`（同上） | **0** | 0 ❌ |
| **20** | `t2,t1` | **`t1c,flair` 都在** | **0** | **0** ❌ |
| 23 | `t2,t1` | `t1c,flair` 都在 | 0 | 0 ❌ |

- **#18 vs #19**：同一 `missing`，一个出结果一个不出 → 与 `missing` 无关；
- **#20/#23**：**两个关键通道都在**却全空 → **直接排除"缺模态"**。

> **反直觉现象（值得警惕）**：#18「只有 `flair` 一个通道」出了 2503 体素，
> 而 #20/#23/#24「`t1c`+`flair` 都在」却是 0 —— **"通道越全反而越差"**不是自然现象，
> 更像是**通道顺序 / 归一化与训练不一致**，或**权重与预处理不配套**。
>
> 另：#18 的 `core=2503, flair=2503` 完全相等是**设计行为** —— `t1c`/`t1` 都缺时
> core 那一路退化成"参考序列"（恰好就是那条 FLAIR），与 flair 路撞同一 UID，
> `task.py` 的守卫把两路统一成同一份掩膜。

### 不重启服务就能诊断：`scripts/probe_goal5.py`

777 例要跑 5.5 小时，重启代价太大 —— 这个探针**独立跑前几例**，不打扰正在运行的进程：

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main
python3 scripts/probe_goal5.py                                                      # 默认取 3 例
python3 scripts/probe_goal5.py /2026aicompetition/datasets/verification/original 5
```

它打印 `missing` / `thr` / `pmax` / `core=终值(阈值以上)` / `flair=…`，
并**直接给出成因判定**（「概率低于阈值」还是「后处理吃掉了」）+ 每例的 `warnings`。

> 会**占用 GPU**（与正在跑的推理共卡），但只跑几例、几十秒。

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
