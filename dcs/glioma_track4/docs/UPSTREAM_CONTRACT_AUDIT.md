# 上游仓库 / 官方契约 · 合规性审计

> 对照对象：上游官方仓库 **`nenudcs/Glioma_recognition`**（`main` 分支）
> 与本地 `ds/GliomaRecognition/dcs/Glioma_recognition-main/`。
>
> 关联文档：
> - [`MODALITY_DIAGNOSTIC.md`](MODALITY_DIAGNOSTIC.md) —— 模态识别的 1 分钟诊断（§1~§1e）
> - [`INFERENCE_EVAL_AND_SUBMIT_GUIDE.md`](INFERENCE_EVAL_AND_SUBMIT_GUIDE.md) —— 推理/测评/提交全链路
> - [`MIGRATE_AND_RESUME_GUIDE.md`](MIGRATE_AND_RESUME_GUIDE.md) —— 迁移与断点续训
> - [`OOM_AND_RESUME_GUIDE.md`](OOM_AND_RESUME_GUIDE.md) —— 内存与续训

---

## 0. 上游仓库是什么（先定预期）

**上游是 P0 骨架，不是可得分实现。**

| 项 | 上游状态 |
|---|---|
| 定位 | 《赛道四自建模型组推理系统》的 **P0 基线**，验证比赛协议与工程链路 |
| Goal1~Goal5 | **全部是 Dummy 零分基线** —— 只能验证"输出格式合规"，不含任何真实模型 |
| Goal3 / Goal4 / Goal5 | 目录里**只有 `model.py` + `task.py`** 两个占位文件 |
| 已有实现 | `app/`（HTTP + callback）、`core/`（配置/注册/runner）、`data/`（NIfTI Loader + 数据结构）、`pipeline/`、`output/`、`observability/`、`scripts/`、`tests/` |
| `data/` 下文件 | **只有 `loader.py` + `structures.py`**（无 `series_selector` / `modality_fallback` / `voxel_modality`） |
| Goal2 | `goal2_duplicate/` 与 `goal2_stitched/` 是**唯一有完整实现**的两个目标 |

> **结论：上游唯一值得当作权威的，是 `README.md` 的「当前规范解释」一节（官方契约原文）
> 与 `data/loader.py`（目录/表读取的行为基线）。模型侧一切都要本地自己实现。**

---

## 1. 官方契约原文（上游 README「当前规范解释」）

逐条摘录，**这是判分的依据**：

| # | 契约原文 | 管什么 |
|---|---|---|
| 1 | 官方输入仅支持 NIfTI（`.nii` / `.nii.gz`） | 数据侧 |
| 2 | 数据根目录可提供标准 `SeriesType.xlsx`，Loader 按 `AccessionNumber` 和 `SeriesUid` 匹配后使用 `SeriesType` 补充序列类型；缺少文件或匹配行时沿用原有元数据 | 表的定位与匹配 |
| 3 | **Series UID 目录包含多个 NIfTI 时，只读取文件主名与目录名完全一致的原文件；没有唯一匹配时明确报错** | 多文件规则 |
| 4 | **核心区写入其来源 T1 增强 Series UID 目录；周围区写入其 Flair/T2 Series UID 目录** | Goal5 掩膜归属 |
| 5 | **`SegmentationMaskURI` 相对于病例目录，格式为 `./{SeriesUid}/{SeriesUid}.nii.gz`** | 掩膜路径 |
| 6 | **所有病例均输出 `IsNotHumanBodyProb` 和 `IsStitchedProb`** | 必填字段 |
| 7 | 二分类字段采用比赛示例中的专属 `*Probability` 名称 | 字段命名 |
| 8 | duplicate JSONL 至少一行；单检查数据集无法同时满足"至少一行"和"禁止 self-pair"，因此会明确失败 | Goal2-B |

---

## 2. 逐条合规性核对（本地实现）

| # | 契约 | 本地实现位置 | 判定 |
|---|---|---|---|
| 1 | 仅 NIfTI | `data/loader.py:25` `_NIFTI_SUFFIXES` | ✅ |
| 2 | 表按 (AccessionNumber, SeriesUid) 匹配 | `data/loader.py:251-360` `_read_series_types` | ✅ **超集**（见 §3） |
| 3 | 多文件只认 `主名 == 目录名`；无唯一匹配报错 | `data/loader.py:123-151` `_select_original_nifti_files` | ✅ 完全一致 |
| 4 | core←T1增强 / peri←Flair·T2 | `tasks/goal5_segmentation/task.py:30-31`：`CORE_SOURCE_MODALITIES=("t1c","t1")`、`FLAIR_SOURCE_MODALITIES=("flair","t2")` | ✅ **已实现** |
| 5 | `./{SeriesUid}/{SeriesUid}.nii.gz` | `pipeline/aggregator.py:8` `MASK_URI_TEMPLATE` | ⚠️ **模板对，但 `{SeriesUid}` 取值有风险 → §4** |
| 6 | 每例都出 `IsNotHumanBodyProb` / `IsStitchedProb` | `pipeline/aggregator.py:30-31`（无条件写入）+ `output/schemas.py` `PREDICTION_REQUIRED_KEYS` | ✅ |
| 7 | 专属 `*Probability` 命名 | `pipeline/aggregator.py:42-69`（`EnhancementProbability` 等） | ✅ |
| 8 | JSONL 至少一行 | `output/writer.py:75-88`（不足 2 例才报错，1 例补齐逻辑） | ✅ |

> 契约 2、3 是本地与上游**行为完全一致**的两处 —— 可以放心。

---

## 3. 本地相对上游的**增强**（超集，不要回退）

上游 `data/loader.py` 的原始行为很"脆"，本地每一条增强都是为了"**一例脏数据不要毁掉整批评测**"：

| 能力 | 上游 | 本地 | 为什么要加 |
|---|---|---|---|
| 表的位置 | **只** `root/SeriesType.xlsx` | + `root/annotation/`、`root/original/`、`root.parent/*` | 实测训练集表在 `annotation/`、验证集在 `original/`，差一层就整表读不到 |
| 表头识别 | 三个**精确**列名 | 别名**子串**（`检查号`/`序列号`/`序列类型`…）+ **取值嗅探** | 官方表中文/英文混着来，列名一改版就 `InvalidInputError` |
| 数据行缺字段 | **抛错 → 整批失败** | 告警**跳过该行**（同一键冲突仍抛错） | 表尾合计/备注行常见，代价与"一行残缺"不相称 |
| 掩膜关键词 | 仅英文 `mask/seg/label/roi` | + 中文 `掩码/标注/瘤体/水肿/…` | 漏检 → 整批失败或静默降级；误检只多读一个文件 |
| 数据集根解析 | 无 | `_resolve_dataset_root`：阶段目录/容器层自动下钻，多候选**响亮报错** | 误指父目录会把阶段名当检查号，静默产出垃圾答案 |
| 单例读取容错 | 无 | `GLIOMA_LOADER_TOLERANT=1` 时跳过坏序列 | 评测**不可重跑**，1 个脏文件 ≠ 整批 0 分 |
| 模态兜底链 | 无 | `series_selector` → `modality_fallback`（官方表）→ `voxel_modality`（体素判别） | 挑不出序列 = Goal5 通道全零 = 该例 0 分 |

**结论：本地的 loader 是上游的严格超集，且每一条增强都有明确的失败场景支撑。迁移时不要退回上游版本。**

---

## 4. ✅ 已修复：掩膜 URI 的 `{SeriesUid}` 取值（唯一高风险不一致）

### 4.1 契约怎么说

契约 5 原文：

> `SegmentationMaskURI` **相对于病例目录**，格式为 `./{SeriesUid}/{SeriesUid}.nii.gz`

而契约 3 与磁盘布局共同定义了 `SeriesUid` 的含义 —— 它就是**磁盘上的那一层目录名**：

```
<病例目录>/<SeriesUid>/<SeriesUid>.nii.gz
              ^^^^^^^^^^^^ = 官方 {SeriesUid} = 磁盘目录名
```

### 4.2 本地链路

```text
Goal5Task._restore(...)
  → infer_uid_from_path(src)                       # tasks/goal5_segmentation/task.py:162
  → Goal5Result.core_source_series_uid
  → PredictionAggregator._mask_uri(uid)            # pipeline/aggregator.py:91
  → "./{uid}/{uid}.nii.gz"
  → OutputWriter._write_masks → <answer>/<acc>/<uid>/<uid>.nii.gz
```

而 `infer_uid_from_path` 的优先级是 **sidecar 优先**：

```141:147:Glioma_recognition-main/data/series_selector.py
def infer_uid_from_path(series) -> str:
    """稳定的序列标识：优先 metadata/UID，退化为目录名。"""
    uid = str((series.metadata or {}).get("SeriesInstanceUID")
              or series.series_uid or "").strip()
    if uid:
        return _sanitize_uid(uid)
    return _sanitize_uid(series.source_path.parent.name)
```

### 4.3 风险推演

**若 sidecar 的 `SeriesInstanceUID` ≠ 磁盘目录名**，则：

| | 值 |
|---|---|
| 磁盘上真实存在的目录 | `<acc>/*2.25.197351…*` |
| Writer 写出的答案目录 | `<answer>/<acc>/2.25.197351…/` ← **少了一对 `*`** |
| 写出的 URI | `./2.25.197351…/2.25.197351….nii.gz` |
| 平台按 `./{SeriesUid}/…` 相对病例目录解析 | ❗ **指向输入数据里不存在的目录** |

**更危险的是：本地 `OutputValidator` 查不出这个问题。**

```73:77:Glioma_recognition-main/output/validator.py
            series_uid = mask_path.parent.name
            try:
                source = study.series_by_uid(series_uid)
            except KeyError:
                self._fail(f"mask URI references unknown series: {uri}")
```

它只用**答案目录里的目录名**去内存里的 `Study` 查（`writer` 与 `validator` 用的是同一个 uid，必然自洽），
**从不比对输入数据的真实目录名** —— 所以本地全绿、平台可能取不到参考几何。

### 4.4 决定性检查（**长度 + 字符码**，绕过 `*` 被 Markdown 吃掉的问题）

> ⚠️ 直接用 `repr()` 粘贴不可靠：`*xxx*` 在 Markdown 里会渲染成斜体、`*` 消失。
> 所以下面用 **`len()` + `ord(首/尾字符)`** 判定 —— `ord('*') = 42`，`ord('2') = 50`。

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 - <<'PY' 2>&1 | tee /tmp/uid_audit.txt
from pathlib import Path

from data.loader import DatasetLoader
from data.series_selector import infer_uid_from_path

ROOT = Path('/2026aicompetition/datasets/verification')
if (ROOT / 'original').is_dir():
    ROOT = ROOT / 'original'
print("ROOT =", ROOT)

n_total = n_ok = 0
bad = []
for i, st in enumerate(DatasetLoader().iter_studies(ROOT)):   # 逐例流式，不同时持有全部影像
    if i >= 40:
        break
    for s in st.series:
        d = s.source_path.parent.name                                   # 磁盘目录名 = 官方 {SeriesUid}
        u = s.series_uid                                                # Study.series_by_uid 的键
        m = str((s.metadata or {}).get('SeriesInstanceUID') or '')       # sidecar 值
        w = infer_uid_from_path(s)                                      # Writer 落盘用的 uid
        n_total += 1
        ok = (d == u) and (d == w)
        n_ok += ok
        if not ok and len(bad) < 8:
            bad.append((st.accession_number, d, u, m, w))
    del st

print(f"\n共检查 {n_total} 条序列：目录名==series_uid==writer_uid 的 {n_ok} 条；"
      f"有不一致的 {n_total - n_ok} 条")
print("KEY: 掩膜 URI 是否安全 ?", "是（无需改代码）" if n_ok == n_total else "否（必须改，见下文）")

def show(tag, v):
    if not v:
        print(f"  {tag:<12} len=  0")
        return
    print(f"  {tag:<12} len={len(v):>4}  first=ord({v[0]!r})={ord(v[0]):>3}  "
          f"last=ord({v[-1]!r})={ord(v[-1]):>3}")

print("\n--- 不一致样例（42 表示该端是星号 `*`，50 表示是数字 '2'）---")
for acc, d, u, m, w in bad:
    print(f"\n[acc={acc}]")
    show("目录名", d)
    show("series_uid", u)
    show("sidecarUID", m)
    show("writer_uid", w)
    print(f"  相等? 目录==series_uid: {d == u} | 目录==writer_uid: {d == w} | "
          f"长度差(dir-uid): {len(d) - len(u)}")
print("\n注：ord('*')=42, ord('2')=50, ord('8')=56")
PY
```

**判读**（契约对齐后的期望是 `n_ok == n_total`）

| 看到 | 结论 | 动作 |
|---|---|---|
| `KEY: 掩膜 URI 是否安全 ? 是（无需改代码）` | `series_uid` 就等于目录名 | ✅ 无需操作 |
| 有 `不一致` 条目 | 该数据的 sidecar UID 与目录名不同 | 本轮已按契约改为**磁盘优先**，应已消除；若仍有 → 贴输出 |

### 4.5 已应用的补丁（**改代码，2 处**）

**决策依据**：按 README「当前规范解释」，`SeriesUid` 的身份由**磁盘目录**定义，
所以优先级必须是「**磁盘推导值 → sidecar**」，而不是反过来。

**改动 1** —— `data/loader.py` 的 `_read_nifti_series`：`Series.series_uid` 改为磁盘优先

```python
# 旧：uid = _clean_identifier(metadata.get("SeriesInstanceUID"), series_uid)   ← sidecar 优先
sidecar_uid = metadata.get("SeriesInstanceUID")
uid = _clean_identifier(series_uid or sidecar_uid, str(sidecar_uid or ""))
```

`series_uid` 在此之前已由 Loader 按布局算好（3 层 → `path.parent.name`，≤2 层 → `_nifti_stem(path)`），
所以这一改**同时**把 `Study.series_by_uid` 的键、掩膜落盘目录名、官方表匹配的候选[0]
全部对齐到契约。sidecar 的值仍完整保留在 `series.metadata["SeriesInstanceUID"]` 里，**无信息损失**。

**改动 2** —— `data/series_selector.py` 的 `infer_uid_from_path`：同样磁盘优先

```python
    uid = str(getattr(series, "series_uid", "") or "").strip()          # = 磁盘推导值
    if uid:
        return _sanitize_uid(uid)
    parent = getattr(getattr(series, "source_path", None), "parent", None)
    if parent is not None and parent.name:
        return _sanitize_uid(parent.name)                               # 回退：目录名
    sidecar = str((getattr(series, "metadata", None) or {}).get("SeriesInstanceUID") or "").strip()
    return _sanitize_uid(sidecar) if sidecar else ""                    # 最后才用 sidecar
```

> **为什么不需要额外改 `Study.series_by_uid`**：改动 1 之后 `series_uid` 恒等于磁盘推导值，
> 于是 `infer_uid_from_path(s) == s.series_uid`（§4.6 的 ②④ 已实测），
> Writer 走的是**精确匹配**，不可能 `KeyError`。
> `core/runner.py::_run_streaming` 对整批只有一层 try（**无 per-case 容错**），
> 所以这里宁可少改一处、也不引入"精确匹配失效"的窗口
> （这正是 `task.py:140-143` 注释里记过的那次事故）。

### 4.6 验证结果（本地，已实跑）

用**真实临时数据集**验证（sidecar 的 `SeriesInstanceUID` 故意与目录名不同，长度差 2 = 一对装饰字符；
`*` 在 Windows 上非法，故本地用 `_` 做等价代理 —— 真实数据在 Linux 容器里是 `*`）：

```text
构造：磁盘目录名 = '_2.25.111111111111111111111111111111111111_'  len=43
      sidecar UID = '2.25.111111111111111111111111111111111111'  len=41

① Series.series_uid == 目录名                              OK
② infer_uid_from_path(s) == 目录名                         OK
③ mask_uri == './<目录名>/<目录名>.nii.gz'                  OK
④ study.series_by_uid(目录名) 找得到（不 KeyError）          OK
⑤ _uid_candidates[0] == 目录名（= 官方表里的 SeriesUid）     OK
⑥ 2 层布局仍走文件主干 'PLAIN'                              OK

结果：9/9 通过 -> 全部符合官方契约
```

回归：`python -m unittest discover -s tests -v` → **`Ran 11 tests ... OK`**
（含 `test_loader_uses_xlsx_series_type_without_replacing_uid` 与 `test_end_to_end` 的掩膜 URI 断言）。

> **仍需在真数据上复核一次**：本地用的是等价代理（`_`），真实目录名带 `*`。
> 请在容器里跑 §4.4，期望 `KEY: 掩膜 URI 是否安全 ? 是（无需改代码）`。
> 若仍有 `不一致` 条目，说明还有第三个来源（例如 `_clean_identifier` 的字符清洗差异），把输出贴出来。

---

## 5. 顺带解决：代码里一个悬而未决的 `【待确认】`

`pipeline/aggregator.py:81-89` 的注释写着：

> 【待确认】组委会材料中"目录规范"与"阳性示例"对掩码路径的描述不一致：
> 目录规范为 `{AccessionNumber}/{SeriesUid}/<series>.nii.gz`，示例却写作 `./output/sub-*_core.nii.gz`。

**上游 README 的官方契约已经把这条钉死了**：

> `SegmentationMaskURI` 相对于病例目录，格式为 `./{SeriesUid}/{SeriesUid}.nii.gz`

即：**目录规范那版是对的，`MASK_URI_TEMPLATE = "./{uid}/{uid}.nii.gz"` 无需改动。**
`aggregator.py` 里那段 `【待确认】` 可以删掉，改为引用官方契约。

---

## 6. 待办清单

- [x] 确认上游是 Dummy 骨架 → 模型侧无可复用实现
- [x] 契约 1/3/6/7/8 逐条核对 → 本地全部满足
- [x] 契约 4（core←T1C、peri←Flair/T2）→ 本地已实现
- [x] 契约 5 的**模板**格式 → 已确认，`aggregator.py` 的 `【待确认】` 可关闭
- [x] 契约 5 的 **`{SeriesUid}` 取值** → **已按 README 改为磁盘优先**（§4.5 改动 1 + 改动 2）
- [x] 本地验证：临时数据集 9/9 通过 + `tests` 11/11 OK（§4.6）
- [ ] **容器里跑 §4.4 复核**：期望 `KEY: 掩膜 URI 是否安全 ? 是（无需改代码）`
- [ ] 收尾跑 `scripts/validate_output.py` 复验答案格式
- [ ] 迁移到原位置后重跑 `python -m unittest discover -s tests -v`

---

## 7. ✅ 已实现：加载容错（正式评测默认开启）

README §3 写的是「Series UID 目录包含多个 NIfTI 时……**没有唯一匹配时明确报错**」，
上游就是 fail-fast。本地多了一个容错开关，**本轮把它从「只有开关」补成了「真正管用」**：

```text
$ ./start.sh                          # 默认已导出 GLIOMA_LOADER_TOLERANT=1
$ GLIOMA_LOADER_TOLERANT=0 ./start.sh # 要严格 fail-fast 就这样起
```

### 7.1 开关覆盖的 4 处（都只在容错模式生效；关闭时与上游完全一致）

| # | 位置 | 默认（fail-fast） | 容错模式 |
|---|---|---|---|
| ① | `data/loader.py::_select_original_nifti_files` | 抛 `InvalidInputError` | 跳过该序列目录 |
| ② | `data/loader.py::DatasetLoader._iter_nifti` | 抛错 | 跳过该序列 |
| ③ | `data/loader.py::_safe_read_series_types`（**本轮新增**） | 抛错 | 整表降级为「无表继续」 |
| ④ | `core/runner.py::_run_streaming`（**本轮新增**） | 单例失败 → `rmtree(staging)`，**整批作废** | 只跳过该例，其余照常发布 |

### 7.2 ③④ 为什么必须补（原来的开关只做了一半）

- `iter_studies` 是**生成器**，`_read_series_types` 的异常在**首次推进时**抛出，
  被 runner 那唯一一层 try 接住 → 整批归零。而「表读不到」恰恰**是有兜底的**：
  描述退回 sidecar / 目录名，随后 `data.modality_fallback` 会用同一张表再兜一次
  （它自带 `try/except` 与告警）。
- runner 的循环原本**没有任何 per-case 保护**：1 例脏数据 = 776 例好数据一起作废。
  而 `OutputValidator.validate_final_layout` 比的是**传进去的成功集合**，
  所以「跳过后发布」在契约上是自洽的（只要把写了一半的该例目录清掉）。

### 7.3 验证（本地，两种模式各跑一遍）

```text
=== fail-fast（GLIOMA_LOADER_TOLERANT 未设 / =0）===
① 脏序列目录            -> 抛 InvalidInputError（exactly one original）     OK
② SeriesType.xlsx 损坏  -> 抛 InvalidInputError（cannot read ...）          OK
③ 单例推理抛异常        -> 抛 RuntimeError 且**未发布**任何目录              OK
                                                      3/3 通过

=== 容错 ON（=1）===
① 脏序列目录            -> 跳过该目录；ACC001 / ACC002(保留 FLAIR) / ACC003 照常  OK
② SeriesType.xlsx 损坏  -> 降级为「无表继续」，仍产出检查                       OK
③ 单例推理抛异常        -> 跳过 ACC002，**发布 ACC001 + ACC003**                OK
                           duplicate_pairs.jsonl 非空                          OK
                           日志含 study_skipped / studies_skipped              OK
                                                      8/8 通过

回归（两种模式各一遍）：python -m unittest discover -s tests
    -> Ran 11 tests ... OK     （两种模式都全绿）
```

### 7.4 顺带修好了 3 个「只在容错模式下会失败」的测试

`tests/test_streaming.py` 有 3 个测试是**断言 fail-fast 语义**的，打开开关后必然失败。
已改为**按模式各自断言正确行为**（不是 skip），所以两种模式下测试都有意义：

| 测试 | fail-fast 断言 | 容错断言 |
|---|---|---|
| `test_loader_rejects_multiple_files_without_exact_original` | 抛 `exactly one original` | 脏目录跳过，同检查下 `SERIES-OK` 照常产出 |
| `test_loader_rejects_conflicting_xlsx_series_types` | 抛 `conflicting SeriesType` | 整表降级，描述退回 `series_uid` |
| `test_failed_study_removes_staging_output` | 抛 `OutputValidationError` | 抛 `ValueError`（无任何一例成功）；**两种模式都不得留下 staging** |

### 7.5 云平台启动方式

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main
./start.sh      # start.sh 内已 : "${GLIOMA_LOADER_TOLERANT:=1}" 并 export
```

> ⚠️ 若你的启动方式**绕过 `start.sh`**（直接 `uvicorn`，或平台的环境变量面板），
> 必须自己保证 `GLIOMA_LOADER_TOLERANT=1` 在**进程启动前**已在环境里 ——
> `_TOLERANT` 是**模块导入期**求值的，跑起来之后再设**不生效**。
> 自检：`python -c "from data.loader import tolerant_mode; print(tolerant_mode())"` 应为 `True`；
> 运行时日志里应出现 `[loader][容错]` / `[runner][容错]` 前缀的告警。
