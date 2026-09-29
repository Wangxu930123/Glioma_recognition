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
| `[selector][模态回退] 挑不出序列，序列表在 .../SeriesType.xlsx 但按 (检查号,序列号) / UID 都匹配不到` | ② 匹配不上 | **§1** |
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
| `命中 UID = 0%` | 文件名不是 `SeriesUid` | 表对不上不奇怪；体素兜底是唯一出路 → **§2** |
| `读到的条数 = 0` | 表没读到或表结构不认 | 检查 §表头原文；或 `export GLIOMA_LABELS_DIR=<表所在目录>` |

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

**先跑 §1**（10 秒）——它会直接区分「表值是 `其他`」和「读错列」这两种完全不同的成因；
**再跑 §2**（40 秒）——它决定体素兜底能不能信（≥80% 可用）；
最后跑完推理看 **§3**，用 `missing_channels` 与掩膜的 `体素和` 判断是否真的救回来了。
三份结论按 **§4** 的表组合，就知道下一步该改什么。
