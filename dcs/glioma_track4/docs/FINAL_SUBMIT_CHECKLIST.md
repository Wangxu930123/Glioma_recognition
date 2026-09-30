# 提交前最终 Checklist

> **定位**：临提交前的作业单。回答三件事 —— **改了什么**、**跑什么**、**看到什么才能交**。
>
> 不覆盖训练与平台界面，那些看：
> - [`INFERENCE_EVAL_AND_SUBMIT_GUIDE.md`](INFERENCE_EVAL_AND_SUBMIT_GUIDE.md) —— 训练 → 推理 → 测评 → 提交全链路
> - [`PLATFORM_GUIDE.md`](PLATFORM_GUIDE.md) —— 云桌面 / 训推平台界面操作
> - [`MODALITY_DIAGNOSTIC.md`](MODALITY_DIAGNOSTIC.md) —— 模态识别与预处理诊断（长文，排障时才翻）
> - [`UPSTREAM_CONTRACT_AUDIT.md`](UPSTREAM_CONTRACT_AUDIT.md) —— 官方契约逐条核对

---

## 0. 三十秒速览

```text
① export 环境变量 + ./start.sh 启动（不要直接 python -m uvicorn）
② python3 scripts/pre_submit_check.py --limit 20     → 等一个 GO
③ GO → 正式提交
```

**三条铁律**

| # | 铁律 | 违反的后果 |
|---|---|---|
| 1 | **必须设 `COMPETITION_PIPELINE_FACTORY`** | 服务静默跑 Dummy 基线，答案全是占位 → 近 0 分（见 §1） |
| 2 | **改了代码必须重启服务** | `Goal5Config` / 后处理 / 容错开关都是进程启动时读取，不重启跑的还是旧代码 |
| 3 | **正式评测别关 `GLIOMA_LOADER_TOLERANT`** | 1 例脏数据 → 整批 staging 被 rmtree → 几百例一起 0 分 |

---

## 1. ⚠️ 头号杀手：`COMPETITION_PIPELINE_FACTORY`

**这是本轮发现的、最容易导致"整批近 0 分"且最难定位的一条。**

`core/registry.py:10-24` 的行为：

```10:24:Glioma_recognition-main/core/registry.py
    if not factory_path:
        # ⚠️ 这段日志必须留着：没有它，Dummy 基线是**静默生效**的 ——
        #    服务照常起来、/health 通、回调也正常，但答案是占位内容。
        print(
            "[registry] ⚠️ 未设置 COMPETITION_PIPELINE_FACTORY → 当前跑的是 **Dummy 基线**"
            ...
        )
        return InferencePipeline()
```

**现象**：服务正常启动、`/health` 通过、回调正常、答案格式完全合规 —— 但内容全是占位。
**难点**：日志里只有一行字，`answer/` 目录看起来也"写好了"。

**已修复**：`start.sh` 现在默认设置它：

```sh
: "${COMPETITION_PIPELINE_FACTORY:=tasks.real_pipeline:build_pipeline}"
export COMPETITION_PIPELINE_FACTORY
```

并额外打印**权重解析结果**（路径写错成旧权重是静默的，所以必须打印）：

```text
[start.sh] 真实插件工厂：tasks.real_pipeline:build_pipeline
[start.sh] 权重根    ：/2026aicompetition/workspace/checkpoint
[start.sh] Goal5 权重：/2026aicompetition/workspace/checkpoint/goal5_segmentation/core.pt
```

**你必须做的**：**用 `./start.sh` 启动**，不要直接 `python -m uvicorn app.server:app`。

> 若要手动启动（**不要**直接 `python -m uvicorn app.server:app`），等价的环境变量是：

```bash
export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
export COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint
export GLIOMA_LOADER_TOLERANT=1
```

---

## 2. 一键预检（唯一必须跑的命令）

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main
python3 scripts/pre_submit_check.py --limit 20 2>&1 | tee logs/pre_submit.log
```

**一条命令、无需参数、退出码即结论**：`0` = GO，`1` = NO-GO。

任一关抛异常都不会让脚本崩 —— 记 FAIL 后继续跑下一关，**一次看到全部问题**，而不是修一个跑一次。

| 参数 | 说明 |
|---|---|
| `--limit N` | 冒烟例数（默认 5；正式提交前建议 20） |
| `--full` | 额外跑完整 runner（Writer+Validator）+ 离线格式校验 —— 提交前最后一道保险，**慢** |
| `--dataset` | 数据根（默认 `/2026aicompetition/datasets/verification/original`） |
| `--train-config` | 训练侧 `preprocess.yaml`（默认已指向 `glioma_track4/configs/preprocess.yaml`） |

**输出长这样**：

```text
========================================================================================
汇总
========================================================================================
  [✔] G1 环境              torch 2.3.0+cu118 / cuda / 1 份权重 / 数据就位
  [✔] G2 权重              seg 头与骨干全部加载成功，权重是训练产物
  [✔] G3 预处理             15 项参数 + 通道链全部与训练侧一致
  [✔] G4 冒烟              空掩膜率 0%（0/20）

========================================================================================
结论：**GO** —— 可以提交
========================================================================================
```

---

## 3. 五关逐关判读

### G1 环境

检查：torch / CUDA / 权重文件可解析 / 数据目录 / **插件工厂变量**。

| 输出 | 判定 | 对策 |
|---|---|---|
| `[✘] COMPETITION_PIPELINE_FACTORY 未设置` | **NO-GO** | 用 `./start.sh` 启动（§1） |
| `[✘] 权重解析失败: FileNotFoundError` | **NO-GO** | 设 `COMPETITION_CHECKPOINT_ROOT`，确认 `<root>/goal5_segmentation/core.pt` 存在 |
| `[✘] 数据目录不存在` | **NO-GO** | `--dataset` 指对路径 |
| `[!] CUDA 不可用` | WARN | CPU 也能跑，但 700+ 例会慢到不可接受 |

### G2 权重

**最容易静默出错的一关。** `inference.load_model` 用 `strict=False` 加载，而缺失键**只过滤
`("enc","dec","bottleneck","stem")` 前缀** —— `seg` 头的键名若对不上会被**静默随机初始化**：
权重校验通过、服务正常启动、阈值照读，而分割输出永远是噪声 → 概率低于阈值 → **空掩膜**。

| 输出 | 判定 | 含义 / 对策 |
|---|---|---|
| `缺失键: 0 个` + `[✔]` | PASS | 权重健康 |
| `!!! seg 头没加载上 → 分割头是随机初始化的` | **NO-GO** | ❗ **权重导出问题**，改预处理救不回来。重跑导出脚本 |
| `state_dict 里没有张量` / `顶层不是 dict` | **NO-GO** | 权重文件不是模型（可能误放了配置/日志） |
| `元数据键: []`（无 `arch`/`thresholds`） | WARN | 不是训练脚本产出 → 阈值走 `(0.5,0.5)` 兜底，分割质量下降 |
| `seg_head.weight: std ≈ 0.10 且 mean ≈ 0` | WARN | 疑似随机初始化（训练后的权重通常 std 0.01~0.3 且均值偏离 0） |

> 需要更细的权重信息（`epoch` / `best_metric` / `fold` / 各折阈值）：
> `python3 scripts/audit_goal5.py`

### G3 预处理

逐项对比 `Goal5Config` 与训练侧 `glioma_track4/configs/preprocess.yaml`。

**这就是本轮空掩膜的主因** —— 修前有 4 处不一致，见 §4。

| 输出 | 判定 |
|---|---|
| `15 项参数 + 通道链全部与训练侧一致` | PASS |
| `N 处与训练不一致: [...]` | **NO-GO** —— 推理输入会超出训练分布 |
| `训练配置不存在 → 无法判定` | WARN —— 用 `--train-config` 指对路径 |

### G4 冒烟

直接对前 N 例推理，**统计空掩膜率**。这是最直观的"能不能交"的指标。

| 空掩膜率 | 判定 | 含义 |
|---|---|---|
| **≤ 20%** | PASS | 可以提交 |
| **20%~60%** | WARN | 可提交但分数受损，建议先看诊断 |
| **> 60%** | **NO-GO** | **先别交** —— 基本必然低分 |

脚本同时打印每例的 `missing` / `pmax` / `thr` / `core(pre)` / `flair(pre)`，用来分造成因：

| 现象 | 成因 | 对策 |
|---|---|---|
| `pmax < thr`（两个通道都是） | **概率低于阈值** —— 模型输出本身不够高 | 通道零占位 / 权重 / 预处理不一致（查 G2、G3） |
| `pre > 0` 而终值 `0` | **后处理吃掉的** | 只有 `min_tumor_voxels=30` 会这样；调小它 |
| `missing` 非空 | 该通道是**零占位** | 检查模态识别；`GLIOMA_VOXEL_GUESS_EXCLUDED` 是否为 1 |

> 更详细的逐例诊断（含每例的序列清单与警告）：
> `python3 scripts/probe_goal5.py /2026aicompetition/datasets/verification/original 5`

### G5 端到端（`--full`）

完整跑一遍 `EvaluationRunner`（Loader → Pipeline → Writer → Validator），再用
`scripts/validate_output.py` 离线校验输出格式。

| 输出 | 判定 |
|---|---|
| `完整链路 + 输出格式校验通过` | PASS |
| `输出格式校验未通过` | **NO-GO** —— 触发平台侧格式错误 |

---

## 4. 本轮修复清单（6 处代码 + 2 个新脚本）

### 4.1 空掩膜的两条主因

| # | 文件 | 问题 | 修法 | 影响面 |
|---|---|---|---|---|
| **1** | `tasks/goal5_segmentation/preprocess.py` | **通道取用链没有 fallback**（缺→零通道），而训练时是**用替代模态顶替**（`t1c←[t1,t2]`、`flair←[t2]`） | 新增 `CHANNEL_FALLBACK`，与训练 yaml 逐字一致 | **~47% 缺 T1CE 的检查**；解释了"通道越全反而越差"的倒挂 |
| **2** | `tasks/goal5_segmentation/config.py` | `max_spacing_factor` 缺失 → 由 `_common/spatial.py` 默认值 **4.0** 兜底，训练侧是 **1.5** | 显式设为 `1.5` 并透传给 `target_grid` | 3mm 层厚的序列：修前被插值到 1mm（`(20,20,30)`），修后保持 3mm（`(20,20,10)`） |
| **3** | `tasks/goal5_segmentation/preprocess.py` | 参考网格优先级 `t1c→t1→flair→t2`，训练侧是 `t1c→**flair**→t2→t1` | `_REF_PRIORITY` 对齐训练 | 缺 T1CE 时两边界出**不同形状与 affine** 的公共网格 |
| **4** | `tasks/goal5_segmentation/config.py` | `overlap = 0.5`，训练侧 `0.4` | 改为 `0.4` | 次要（滑窗接缝） |

> **1 的实测差异**（只有 `T2WI` + `T2-Flair` 的检查）：
> ```text
> 修前: missing = ('t1c', 't1')   非零通道 2 个   ← t1c 通道全零
> 修后: missing = ('t1',)         非零通道 3 个   ← t1c ← t2 顶替
> ```

### 4.2 官方契约对齐

| # | 文件 | 问题 | 修法 |
|---|---|---|---|
| **5** | `data/loader.py`、`data/series_selector.py` | `SeriesUid` 是 **sidecar 优先**，而官方 README「当前规范解释」把它定义为**磁盘目录名** | 改为**磁盘推导优先**，`SeriesInstanceUID` 只兜底 |
| **6** | `pipeline/aggregator.py` | `MASK_URI_TEMPLATE` 与 `_mask_uri` 里挂着"目录规范 vs 阳性示例不一致，待组委会确认"的 `【待确认】` | 换成官方契约引用（`./{SeriesUid}/{SeriesUid}.nii.gz`），三种猜测收敛为一种 |

> **为什么 5 重要**：sidecar 与目录名不一致时会写出一个**输入数据里不存在**的目录。
> 而本地 `OutputValidator` 只拿**答案目录名**去内存里的 `Study` 查（Writer/Validator 共用同一个 uid，
> **必然自洽**）—— **本地全绿、平台取不到参考几何**。

### 4.3 后处理

| # | 文件 | 问题 | 修法 |
|---|---|---|---|
| **7** | `tasks/goal5_segmentation/postprocess.py`、`task.py` | `clean_pair` **没传 `spacing`** → `bridge_mm=10.0` 被按 1mm 硬算，结构元恒为 **21³** | 透传 `spacing_of(prepared.affine)` |

**后果量化**：

| 层厚 | 应有半径 | 修前实际 | 放大 |
|---|---|---|---|
| 1mm | 10 体素 | 10 体素 | 1× |
| **3mm** | **3 体素** | **10 体素** | **3.3×** |
| 5mm | 2 体素 | 10 体素 | 5× |

闭运算会把**远离病灶的假阳性斑点**与病灶连成一体 → 拉低 Precision 与 HD95。
**这个 bug 不报错、不变空、只悄悄掉分**，所以特别隐蔽。

> **后处理里唯一会让非空掩膜变空的代码**（已逐行确认）：
> ```57:57:Glioma_recognition-main/tasks/goal5_segmentation/postprocess.py
>     return m if int(m.sum()) >= int(min_voxels) else np.zeros_like(m, dtype=bool)
> ```
> `_largest_components` 与 `_bridge` 都是**扩张性**操作，不可能把非空掩膜变空。
> **推论**：诊断里 `core_pre_voxels = 0` ⟹ 必然是"概率低于阈值"，不是后处理吃掉的。

### 4.4 启动脚本

| # | 文件 | 改动 |
|---|---|---|
| **8** | `start.sh` | 显式设置 `COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline`（见 §1），并打印权重解析结果 |

### 4.5 新增脚本

| 脚本 | 用途 |
|---|---|
| **`scripts/pre_submit_check.py`** | 提交前一键预检，GO / NO-GO（§2） |
| **`scripts/audit_goal5.py`** | 权重真伪 + 预处理一致性的**逐项**审计（比预检更详细） |

**两个脚本都已实跑验证**（合成权重 + 合成数据）：

```text
pre_submit_check  GO 路径  : G1✔ G2✔ G3✔ G4✔（空掩膜率 0%）→ GO，退出码 0
pre_submit_check  失败路径 : seg 头键名改坏 → 「缺失键 2 个 {'seg_head': 2}」
                            「!!! seg 头没加载上 → 分割头是随机初始化的」→ NO-GO，退出码 1
pre_submit_check  工厂关卡 : 未设 → FAIL；已设 → PASS
```

---

## 5. 环境变量清单

### 5.1 `start.sh` 已自动设置（默认值即可）

| 变量 | 默认 | 含义 |
|---|---|---|
| `GLIOMA_LOADER_TOLERANT` | **1** | 逐例容错。**1 例脏数据 ≠ 整批作废**（见 §0 铁律 3） |
| `GLIOMA_VOXEL_GUESS_EXCLUDED` | **0** | **不**用体素判别覆盖官方表的「其他」。表里写 `其他` 是**权威排除**，硬猜的代价（错填 OOD 输入）大于收益（留空） |
| `GLIOMA_MODALITY_MODEL` | `glioma_track4/data/modality_model.json` | 重训后的模态判别模型。**缺失时退回内嵌系数（实测一致率约 27%，低于随机 33%）**，`start.sh` 会大声报警 |
| `COMPETITION_PIPELINE_FACTORY` | `tasks.real_pipeline:build_pipeline` | **本次新增**，见 §1 |

### 5.2 可能需要你显式设置

| 变量 | 默认 | 何时需要改 |
|---|---|---|
| `COMPETITION_WORKSPACE` | `/2026aicompetition/workspace` | 路径与规范不同时 |
| `COMPETITION_CHECKPOINT_ROOT` | `{workspace}/checkpoint` | 权重不在规范位置时 |
| `GLIOMA_GOALS` | 全开（6 个） | 只想启用部分插件时，如 `goal5` |
| `GLIOMA_DEVICE` | `cuda` | 强制 CPU 调试时设 `cpu` |
| `GLIOMA_PROGRESS` | 开 | 设 `0` 关闭逐例进度日志 |

**一段可整段粘贴**：

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main

export COMPETITION_WORKSPACE=/2026aicompetition/workspace
export COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint
export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
export GLIOMA_LOADER_TOLERANT=1
export GLIOMA_VOXEL_GUESS_EXCLUDED=0
export GLIOMA_PROGRESS=1

ls -l "$COMPETITION_CHECKPOINT_ROOT/goal5_segmentation/" || echo "!! 权重不在这个位置"
```

---

## 6. 故障速查

| 现象 | 最可能成因 | 命令 / 动作 |
|---|---|---|
| 服务起来了、日志只有一行 `⚠️ 未设置 COMPETITION_PIPELINE_FACTORY` | 跑了 Dummy 基线 | 用 `./start.sh` 启动（§1） |
| `goal5{core=0, flair=0}`，`pmax` 只有 0.2~0.3 | 预处理与训练不一致 / seg 头随机初始化 | `pre_submit_check.py` 的 G2、G3 |
| `goal5{core=0, flair=0}`，但 `pmax > thr` | 后处理吃掉了 | `min_tumor_voxels` 调小（§3 G4 表） |
| 空掩膜率高，且 `missing` 里总有 `t1c` | 缺 T1CE 的检查被零占位 | 确认已同步本轮修复（§4.1 #1） |
| 掩膜偏大、Dice 低、HD95 差 | `bridge_mm` 桥接半径被放大 | 确认已同步本轮修复（§4.3 #7） |
| `FileNotFoundError` 找不到权重 | `COMPETITION_CHECKPOINT_ROOT` 不对 | `ls <root>/goal5_segmentation/core.pt` |
| `OutputValidationError: mask URI references unknown series` | `SeriesUid` 与磁盘目录名不一致 | 确认已同步本轮修复（§4.2 #5） |
| 1 例失败后整批消失（staging 被删） | 容错开关被关 | `GLIOMA_LOADER_TOLERANT=1`（§0 铁律 3） |
| 体素判别把 `其他` 序列当成 T1CE | 覆盖开关被打开 | `GLIOMA_VOXEL_GUESS_EXCLUDED=0` |
| 日志停住不动，分不清卡死还是慢 | 进度日志被关 | `GLIOMA_PROGRESS=1` |

---

## 7. 提交顺序（照抄）

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main

# ---------- ① 环境变量 ----------
export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
export COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint
export GLIOMA_LOADER_TOLERANT=1
export GLIOMA_VOXEL_GUESS_EXCLUDED=0

# ---------- ② 一键预检（看到 GO 再往下）----------
mkdir -p logs
python3 scripts/pre_submit_check.py --limit 20 2>&1 | tee logs/pre_submit.log
# 退出码 0 = GO / 1 = NO-GO
echo "退出码 = $?"

# ---------- ③ NO-GO 时：跑详细探针定位 ----------
python3 scripts/audit_goal5.py 2>&1 | tee logs/audit_goal5.log
python3 scripts/probe_goal5.py /2026aicompetition/datasets/verification/original 5

# ---------- ④ GO 之后：重启服务（必须！改了代码不重启跑的还是旧代码）----------
pkill -f "uvicorn app.server" || true
sleep 2
nohup ./start.sh > logs/server.log 2>&1 &
sleep 8
grep -E "真实插件工厂|Goal5 权重|真实插件 [0-9]/|Dummy 补位" logs/server.log

# ---------- ⑤ 最后一遍完整端到端（慢，但这是最后一道保险）----------
python3 scripts/pre_submit_check.py --limit 20 --full 2>&1 | tee -a logs/pre_submit.log

# ---------- ⑥ 正式提交 ----------
# 界面路径见 PLATFORM_GUIDE.md
```

**第 ④ 步的期望输出**（缺任何一条都说明没跑真实插件）：

```text
[start.sh] 真实插件工厂：tasks.real_pipeline:build_pipeline
[start.sh] Goal5 权重：/2026aicompetition/workspace/checkpoint/goal5_segmentation/core.pt
[real_pipeline] GLIOMA_GOALS=goal1,goal2_duplicate,goal2_stitched,goal3,goal4,goal5 → 真实插件 6/6: ...  ✓ 无 Dummy 补位
[registry] ✓ 真实插件工厂已加载: tasks.real_pipeline:build_pipeline
```

> ⚠️ 看到 `真实插件 N/6` 里 **N < 6** 或 `⚠️ Dummy 补位: ...` —— 那几项还是假的，
> 用 `GLIOMA_GOALS` 确认你想启用哪些。

---

## 8. 回滚与兜底

| 场景 | 动作 |
|---|---|
| 预检 NO-GO 但时间不够 | **先交仍然"合规"的那一版**：只要 G1+G2+G3 通过，答案格式就是合规的（有效分而非 0 分）。G4 空掩膜率高只是分数低，不会作废 |
| 新权重有问题 | 回退到上一份 `core.pt`，重跑 §7 ② |
| 需要严格对齐上游 fail-fast | `GLIOMA_LOADER_TOLERANT=0 ./start.sh`（**不建议**在正式评测用，见 §0 铁律 3） |

---

## 9. 相关文档索引

| 文档 | 什么时候看 |
|---|---|
| [`INFERENCE_EVAL_AND_SUBMIT_GUIDE.md`](INFERENCE_EVAL_AND_SUBMIT_GUIDE.md) | 训练 → 推理 → 测评 → 提交全链路（含打快照、阈值标定） |
| [`PLATFORM_GUIDE.md`](PLATFORM_GUIDE.md) | 平台界面零经验版 |
| [`TRACK4_RUNBOOK.md`](TRACK4_RUNBOOK.md) | 赛道四总作业单 |
| [`MODALITY_DIAGNOSTIC.md`](MODALITY_DIAGNOSTIC.md) | 模态识别 / 预处理 / 后处理诊断（§1~§3f） |
| [`UPSTREAM_CONTRACT_AUDIT.md`](UPSTREAM_CONTRACT_AUDIT.md) | 官方契约逐条核对与修复记录 |
| [`DATASET_ROOT_TROUBLESHOOT.md`](DATASET_ROOT_TROUBLESHOOT.md) | 数据路径找不到时 |
| [`OOM_AND_RESUME_GUIDE.md`](OOM_AND_RESUME_GUIDE.md) | 显存不足 / 断点续训 |
