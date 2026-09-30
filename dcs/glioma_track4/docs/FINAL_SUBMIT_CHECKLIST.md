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
  [✔] G6 内存              常驻模型 5 个（每 Goal 单份）
  [✔] G7 后处理            真实体积下 _bridge 241 MB / 0.56s

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

逐项对比**两份**推理侧预处理与训练侧 `glioma_track4/configs/preprocess.yaml`：

- `channels.*` —— Goal5 那份（`tasks/goal5_segmentation/preprocess.py`）
- `shared.*` —— **共享骨干那份**（`tasks/_common/volume.py`，Goal1/2/3/4 走它）

**这就是本轮空掩膜的主因** —— 修前有 4 处不一致；共享那份还额外漏了 3 处，见 §4.1、§4.9。

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

### G6 内存（常驻模型总数）

算出**会有多少个模型常驻** —— 这是"评测一启动容器就 OOM"的一类成因。

```text
常驻模型总数 = 不同 ckpt_rel 的个数 × 每个 ckpt_rel 解析出的 *.pt 份数
```

| 输出 | 判定 | 对策 |
|---|---|---|
| 每个 Goal 都是 1 份、总数 ≤ 8 | PASS | — |
| 有目录**解析出多份**（`← 目录里没有 core.pt`） | WARN | 确认是有意的多折，而不是训练快照混在里面（§4.7） |
| 总数 > 8 | WARN | 在容器里跑 `scripts/mem_audit.py --trace` 量峰值 |

### G7 后处理（**真实体积**下的峰值内存 / 耗时）

用 `240×240×155`（真实 1mm 脑尺寸）跑一次 `_bridge`，量峰值内存与耗时。
**这一关守卫的正是本次 OOM 的真正元凶**（§4.8）：`_bridge` 曾用 21³ 稠密结构元，
单次 `+701 MB / 31.7 s`，而 `clean_pair` 会调用它两次。

| 输出 | 判定 | 说明 |
|---|---|---|
| `_bridge 241 MB / 0.56s` | PASS | 当前实现（EDT + 包围盒裁剪） |
| `峰值 >550 MB` | **NO-GO** | ❗ 极可能退回了稠密结构元 → 评测必 OOM |
| `耗时 >5 s` | **NO-GO** | 一例拖到分钟级 → 超时 |

> 已验证守卫真的有效：把 `_bridge` 换回旧实现 → G7 报
> `_bridge 单次峰值 703 MB（>550MB）→ 极可能退回了稠密结构元，评测时会把容器 OOM-kill` → **NO-GO**。

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
| **`scripts/mem_audit.py`** | 内存审计：集成规模行为 + 单份模型常驻 + **逐阶段轨迹**（`--trace`）（§4.7） |
| **`scripts/memprobe.py`** | pytest 内存探针（`-p scripts.memprobe`），用来证伪"测试套件是 OOM 元凶"（§4.7） |
| **`scripts/stage_mem_trace.py`** | **逐阶段追踪器**：容器被 OOM-kill 时日志最后一行 `→` 即元凶阶段；`--goals goal5` 可二分定位（§4.10） |

**两个脚本都已实跑验证**（合成权重 + 合成数据）：

```text
pre_submit_check  GO 路径  : G1✔ G2✔ G3✔ G4✔（空掩膜率 0%）→ GO，退出码 0
pre_submit_check  失败路径 : seg 头键名改坏 → 「缺失键 2 个 {'seg_head': 2}」
                            「!!! seg 头没加载上 → 分割头是随机初始化的」→ NO-GO，退出码 1
pre_submit_check  工厂关卡 : 未设 → FAIL；已设 → PASS
```

### 4.6 契约测试的环境依赖（本轮修复）

| # | 文件 | 问题 | 修法 |
|---|---|---|---|
| **9** | `tests/contracts/test_goal5_contract.py` | `test_task_does_not_touch_answer_dir` **只钉了 `COMPETITION_WORKSPACE`，没钉 `COMPETITION_CHECKPOINT_ROOT`** → 容器里有权重时模型真的被加载，`predict` 不抛 `FileNotFoundError` → `Failed: DID NOT RAISE` | 在测试里把权重根也钉进临时目录 |

**症状很迷惑**：CI 上通过（那里没有权重根），一进容器就失败。

**根因是测试依赖环境，不是被测代码有问题** —— 文件头自己写着"不依赖真实训练权重，因此可在 CI 中稳定运行"，
但这一条恰好依赖了"环境里没有权重根"。实证：

```text
场景1（只钉 WORKSPACE，修改前）
  COMPETITION_CHECKPOINT_ROOT = .../ck_evidence（有真权重）
  → predict 正常返回 → **没有抛 FileNotFoundError**       ← 正是你看到的失败
  → （模型真的加载了：.../ck_evidence/goal5_segmentation/core.pt）

场景2（同时钉死 CHECKPOINT_ROOT，修复后）
  → 抛 FileNotFoundError: 未找到权重：.../no-such-ckpt/...   ✓ 与外部环境无关
```

**修法**：

```python
monkeypatch.setenv("COMPETITION_WORKSPACE", str(tmp_path / "ws"))
# 钉死权重根 → 本例内**必然**找不到权重，与外部环境无关
monkeypatch.setenv("COMPETITION_CHECKPOINT_ROOT", str(tmp_path / "no-such-ckpt"))
```

**验证（两种环境都跑）**：

```text
A. 环境里有真权重（复现容器场景）: python -m pytest tests/contracts -q  → 55 passed
B. 环境里没有权重（CI 场景）    : python -m pytest tests -q            → 66 passed
```

> 同仓还有一处**相同模式**（`test_study_tasks_contract.py::test_build_task_has_no_answer_side_effect` 也只钉了
> `WORKSPACE`），但它只调 `build_task`、**不加载模型**，实测两种环境都通过，因此未改动。
> 若将来某个 Goal 的 `build_task` 改成"构造即加载权重"，它就会变成同一个坑。

### 4.7 内存 / OOM（修 2 处放大点 + 新增 2 个诊断工具）

> ⚠️ **OOM 的真正元凶在 §4.8**（`_bridge` 的稠密结构元，只有"空掩膜→非空"之后才会执行）。
> 本节讲的是另外两处**放大点**与诊断工具 —— 它们会让问题更严重，但不是这次的根因。

**先给实测结论，避免误判方向**：

```text
python -m pytest tests -p scripts.memprobe -q
  → 66 passed，峰值 RSS = 359 MB，无任何单个测试增量 > 20 MB
```

**→ 测试套件本身不会 OOM 容器。** OOM 的成因在别处：**权重目录内容**或**推理期峰值**。

#### 修的两处放大点

| # | 文件 | 问题 | 修法 |
|---|---|---|---|
| **10** | `tasks/goal5_segmentation/inference.py` | `resolve_ckpts` 找不到规范文件名时，把目录下**全部** `*.pt` 当集成成员；而 `BackboneRunner.load()` 把**每一份**都构建成**常驻模型**。混入训练快照 → 份数翻倍，且快照带 Adam 优化器状态（单份体积是推理权重的数倍） | 加 `GLIOMA_MAX_ENSEMBLE`（默认 **8**）上限；超限**明确报错**并列出全部文件与三种修法 |
| **11** | `tasks/_common/backbone_runner.py` | 每个 `SingleHeadStudyTask` 各建一个 `BackboneRunner`，而基类 `ckpt_rel` 的**默认值就是同一个文件** → goal1/2/3/4 把**同一份权重加载 4 遍**，全程无共享 | 新增进程级 `_SHARED_WEIGHTS`：同 `(ckpt_root, ckpt_rel, in_ch, device)` 只加载一次，多个任务共享同一批模型对象 |

**常驻模型数的公式（修前 → 修后）**：

```text
修前：启用 Goal 数        × 每个 ckpt_rel 解析出的 *.pt 份数
修后：不同 ckpt_rel 的个数 × 每个 ckpt_rel 解析出的 *.pt 份数
```

**诚实说明**：本仓骨干仅 **1.33 M 参数（5.3 MB fp32）**，所以单纯"多折"通常吃不爆内存 ——
真正危险的是**目录里混进训练快照**（`epoch_*.pt` / `last.pt` / 备份），
它们既让份数翻倍、单份体积又是推理权重的数倍，两个因子相乘才是 OOM 的量级。

**权重复用实测**（4 个 runner，`ckpt_rel` 相同）：

```text
[ckpt] 加载权重 goal5_segmentation/core.pt：1 个成员（存入进程级共享缓存…）
[ckpt] 复用已加载权重（…）    ← runner#1
[ckpt] 复用已加载权重（…）    ← runner#2
[ckpt] 复用已加载权重（…）    ← runner#3
4 个 runner 是否共享**同一个模型对象**：True
shared_weight_count = 1（换成不同 ckpt_rel 后 = 2）
```

**集成上限实测**（3 条分支全过）：

```text
目录里 5 个 fold*.pt（≤ 8）      → 正常集成，打印 [ckpt] …加载全部 5 份
目录里 12 个 *.pt（> 8）         → ValueError，列出全部文件 + 三种修法
GLIOMA_MAX_ENSEMBLE=12 后再跑     → 放行
```

#### 两个诊断工具

```bash
# ① 预检新增 G6：直接算出**常驻模型总数**
python3 scripts/pre_submit_check.py --limit 20

# ② 逐阶段内存轨迹（在容器里跑，定位峰值出现在哪一步）
python3 scripts/mem_audit.py --trace --dataset <数据根>

# ③ 证伪"测试套件是元凶"
python3 -m pytest tests -p scripts.memprobe -q
```

**G6 的输出**（实测）：

```text
  goal5              rel=goal5_segmentation/core.pt         解析出 1 份
  goal3              rel=goal3_tumor/model.pt               解析出 3 份  ← 目录里没有 model.pt
不同 ckpt_rel 数 = 2 | **常驻模型总数 = 4**（已按进程级权重复用折算）
```

**`mem_audit.py --trace` 的判读表**：

| 峰值出现在 | 成因 | 对策 |
|---|---|---|
| 1 加载 Study | 单例影像过大 | 检查原始分辨率 / 层数 |
| 2 `build_volume` | 1mm 公共网格重采样（`D×H×W×4 通道×4B`） | 正常；不同 `ckpt_rel` 的每个 Goal 会各建一次 |
| **3 `load_model`** | **常驻模型数 = 不同 ckpt_rel 数 × 份数** | 清掉目录里多余 `*.pt`，或只留 `core.pt` |
| 4 `infer_segmentation` | 滑窗累加器 + TTA | 调小 `patch` / `tta_batch` |

#### 容器里排查 OOM 的三步

```bash
# ① 权重目录里到底有多少份？（每一份都会变成常驻模型）
for d in "$COMPETITION_CHECKPOINT_ROOT"/*/; do
  echo "$(ls -1 "$d"*.pt 2>/dev/null | wc -l) 份  $d"
done

# ② 常驻模型数 + 逐阶段峰值
python3 scripts/pre_submit_check.py --limit 0        # 看 G6 那一段
python3 scripts/mem_audit.py --trace                 # 看峰值落在哪一步

# ③ 先证伪"测试套件是元凶"
python3 -m pytest tests -p scripts.memprobe -q       # 期望 峰值 < 500 MB
```

### 4.8 ⚠️ OOM 的**真正元凶**：`_bridge` 的稠密结构元（**已修**）

> 你给的因果关系是**对的**，而且正好指向它：**空掩膜时代不 OOM，修完空掩膜就 OOM。**

原因在后处理开头那两行判断：

```51:57:Glioma_recognition-main/tasks/goal5_segmentation/postprocess.py
    m = np.asarray(prob) > float(threshold)
    if not m.any():
        return m.astype(bool)          # ← 空掩膜在这一行就返回了
    if bridge_mm > 0:
        m = _bridge(m, bridge_mm, spacing)      # ← **只有非空掩膜才会走到这里**
```

**空掩膜时代 `_bridge` 从未执行过。** 而它用的是 **21³ 稠密结构元**的
`ndimage.binary_closing` —— 掩膜一旦变成非空，这一段就第一次真正跑起来了。

**实测**（`240×240×155` = 8.9 M 体素，就是真实 1mm 脑的大小）：

| 场景 | 旧实现（21³ 稠密结构元） | 新实现（EDT + 包围盒） | 改善 |
|---|---|---|---|
| **1mm 网格（最常见）** | **+701 MB / 31.7 s** | **+112 MB / 0.30 s** | **省 589 MB、快 104×** |
| 3mm 层厚 | +93 MB / 10.6 s | +119 MB / 0.29 s | 快 37× |

输出一致性 **IoU = 0.9961 / 1.0000**（形状几乎不变）。

而 `clean_pair` 会调用它**两次**（core + flair），所以**一例**的瞬时峰值：

```text
旧：~1.4 GB 峰值 / 63 s   → 真实体积下 binary_closing 的内存/耗时随体积**超线性**增长 → OOM-kill
新：  225 MB 峰值 / 0.83 s（端到端 clean_pair 实测）
```

#### 修法（三层，缺一不可）

1. **欧氏距离变换替代稠密卷积**
   闭运算 = `edt(dilate(mask)) > r`，其中 `dilate(mask) = edt(~mask) <= r`。
   内存 O(体积)、耗时 O(体积)（精确 EDT，不是暴力卷积）。
2. **只在掩膜包围盒外扩 `r` 的窗口内计算**
   闭运算 ⊆ 膨胀 ⇒ "距掩膜 > `r`"的体素恒为背景 ⇒ 窗口外结果就等于输入。
   这一步让内存与**整脑大小基本无关**（`+701 MB → +112 MB` 主要来自它）。
3. 附带正确性收益：距离变换按 `spacing` 取样得到**各向同性球**，而稠密立方结构元是
   **各向异性**的（3mm 层厚轴上半径被放大 3 倍）。现在与"`bridge_mm` 按 mm 给定"的语义一致。

#### 新增契约测试锁定（防回归）

`tests/contracts/test_goal5_contract.py::test_bridge_matches_closing_and_stays_local`：

- 输出与闭运算参考实现 **IoU > 0.9**；
- **距掩膜 > radius 的体素必须保持背景** ← 这条同时保证"包围盒裁剪"这个内存优化是安全的；
- 空掩膜 / `radius <= 0` 直接返回，不进距离变换。

**另外加了自动守卫 G7**（`scripts/pre_submit_check.py`）：用 `240×240×155` 跑一次
`_bridge` 并按**峰值内存/耗时**判定。之所以必须有这条，是因为**单元测试测不到它** ——
仓里的测试都用 `4³`~`8³` 小体积，从来不会触发内存问题；只有真实尺寸才暴露。
阈值 `>550 MB` 或 `>5 s` 判 NO-GO，实测能把两者稳稳分开：

```text
当前实现（EDT + 包围盒）: G7 = PASS   _bridge 241 MB / 0.56s
换回旧稠密结构元（模拟回归）: G7 = FAIL   _bridge 单次峰值 703 MB（>550MB）→ 评测时会把容器 OOM-kill
```

> **为什么之前一直没暴露**：这个 bug **不报错、不变空、只吃内存和时间**，
> 而空掩膜时代它压根不执行 —— 典型的"修好 A 才暴露 B"。
>
> 这也解释了 §4.7 里那句"测试套件不是元凶"为什么成立：
> **pytest 里没有任何测试触碰真实整脑的 `_bridge`**（测试用的都是 `4³`~`8³` 的小体积），
> 所以它从来没在 CI 里被量出来过。

### 4.9 训练 ↔ 推理 预处理一致性（**本轮补上第二份实现**）

#### 训练的事实来源（有代码依据，不是推测）

```212:217:Glioma_recognition-main/tasks/_common/training/helpers.py
    root = _load_track4()
    ds_mod = __import__("src.data.dataset", fromlist=["*"])
    utils = __import__("src.utils.config", fromlist=["*"])
    pre = utils.load_config("preprocess.yaml")
```

即 **`glioma_track4/configs/preprocess.yaml` + `src/data/dataset.py::GliomaDataset`**。

#### 推理侧有**两份**预处理 —— 这是关键

| 实现 | 谁在用 |
|---|---|
| `tasks/_common/volume.py` | **Goal1 / Goal2-stitched / Goal3 / Goal4**（共享骨干 `BackboneRunner`） |
| `tasks/goal5_segmentation/preprocess.py` | **Goal5**（它走滑窗，不参与共享骨干） |

**问题**：上一轮只对齐了 Goal5 那一份 —— 结果 Goal5 空掩膜修好了，
而 **Goal1/2/3/4 四个头继续吃 OOD 输入**（真实性概率、拼接概率、肿瘤概率、14 个结构化字段 + 嵌入）。

#### 本轮补上的三处对齐（`tasks/_common/volume.py`）

| 项 | 训练侧 | 修前 | 修后 |
|---|---|---|---|
| **通道取用链** | `t1c←[t1,t2]`、`flair←[t2]`、`t2←[]`、`t1←[]` | **无 fallback**（缺→零通道） | 与训练**逐字一致** |
| **参考网格优先级** | `t1c→flair→t2→t1` | `t1c→**t1**→flair→t2` | `t1c→flair→t2→t1` |
| **`max_spacing_factor`** | **1.5** | **4.0**（`target_grid` 默认值兜底） | **1.5** |

与 Goal5 那三处（§4.1 #1/#2/#3）**完全同源** —— 因为是同一个成因。

#### 验证（11/11 通过）

```text
A. 常量对账（与 preprocess.yaml / dataset.py 源码逐条比）
  训练 yaml 取用链 : {'t1c': ('t1c','t1','t2'), 'flair': ('flair','t2'),
                     't2': ('t2',), 't1': ('t1',)}
  [OK] 两份推理侧 CHANNEL_FALLBACK 与训练 yaml 逐字一致
  [OK] 两份推理侧 max_spacing_factor 与训练一致（1.5）
  [OK] 两份推理侧参考网格优先级与训练一致（('t1c','flair','t2','t1')）

B. 功能验证（只给 T2WI + T2-Flair 的检查）
  [tasks/_common/volume.py]  missing=('t1',)
    通道来源 = {'t1c': 't2', 'flair': 'flair', 't2': 't2'}   网格 (20,20,10)
  [goal5/preprocess.py]      missing=('t1',)  相同的来源与网格
  [OK] 两份实现的 volume **逐体素相同**（np.allclose）
  [OK] 3mm 层厚轴保持原样（max_spacing_factor=1.5 生效）
```

#### 新增契约测试（防"只修一半"）

`tests/contracts/test_goal5_contract.py::test_shared_and_goal5_preprocess_agree_with_training`
锁死三件事：两份互相同源、通道链与训练 yaml 逐字一致、参考序与 `max_spacing_factor` 与训练一致。

> 预检 **G3** 现在也**两份一起核对**（输出里 `channels.*` 是 goal5 那份、`shared.*` 是共享骨干那份）。

#### 另一工程副本已同步

`glioma_goals/shared/volume.py` 是**同一份代码**（另一工程用它），已同步这三处
—— 审计要求两工程 `volume.py` "实现同源"。

#### 一处**刻意保留**的差异（已确认是训练仓自己的设计）

`global_view` 的中心点：训练用**病灶质心**（有掩码），推理用**前景包围盒中心**。
训练仓**自己**提供了推理侧的对应实现并注明原因：

```559:565:glioma_track4/src/data/dataset.py
def brain_center(vol: np.ndarray) -> np.ndarray | None:
    """非零体素包围盒中心（推理时无掩码可用）。"""
    nz = np.argwhere(np.abs(vol).sum(0) > 1e-3)
```

推理侧 `global_view(center=None)` 与它**公式逐字一致**（`abs(vol).sum(0) > 1e-3` → 取包围盒中点）✓
`size_mm=192 / out=96` 也与训练 `global_view` 一致 ✓。
这是**无法避免的近似**（推理时确实没有掩码），不是 bug，**不要"修"** ——
推理侧改成任何别的中心反而会加重分布偏移。

> 另外：`zscore` 与训练 `zscore_volume` 是**等价实现**（§3e 已逐行对比），无需改动。

### 4.10 ⚠️ "第一例就 OOM-kill 容器"的定位与防护（32GB 容器）

**先分清两类问题**（32GB 被**第一例**打爆 = 瞬时峰值，不是累积泄漏）：

| 类型 | 表现 | 定位工具 |
|---|---|---|
| **累积泄漏** | 越跑越涨、最后几例才死 | `python -m pytest tests -p scripts.memprobe -q`（实测**无泄漏**：后半段 +0.02 MB/测试） |
| **瞬时峰值** | **第一例就死** ← 你的情况 | `scripts/stage_mem_trace.py`（本轮新增） |

#### ① 逐阶段追踪器（**容器死前最后一行日志 = 元凶阶段**）

```bash
cd /2026aicompetition/workspace/dcs/GliomaRecognition/dcs/Glioma_recognition-main
export COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint
python3 scripts/stage_mem_trace.py 3 2>&1 | tee logs/stage_trace.log
# 二分定位：python3 scripts/stage_mem_trace.py 3 --goals goal5
```

每个阶段**进入时立即打印** `→`（flush）—— 容器被 OOM-kill 的那一刻，**日志最后一行 `→` 就是正在执行的阶段**：

| 日志停在 | 元凶 | 下一步 |
|---|---|---|
| `→ load <acc>` | **该例原始 NIfTI 过大** | 用 nib 打印该例各序列的 shape/dtype/大小 |
| `→ <acc> goal5` | 公共网格爆炸 / 滑窗 / EDT（护栏会先报，见 ②） | 看此行前 build_volume 的报错 |
| `→ <acc> goal1/2/3/4` | `global_view` 或常驻模型数 | 跑 G6；看 `real_pipeline` 启动行 |
| `→ build_pipeline` | **权重本身就装不下**（目录 `*.pt` 太多） | 清目录（§4.7） |

**本机端到端实测输出**（真实插件 + 全部权重 + CPU）：

```text
→ build_pipeline（加载全部权重）
← build_pipeline    rss 423 MB (+55)              ← 6 个常驻模型 = shared_weight_count 4+…
→ load ACC0001（2 序列）
→   ACC0001 goal1
←   ACC0001 goal1   rss 1915 MB (+1492)  峰值 2830 MB   ←← 一次性 lazy-init（见 ③）
→   ACC0001 goal2_stitched
←   ACC0001 goal2_stitched  rss 1916 MB (+1)     ← 前向缓存命中：只 +1MB
→   ACC0001 goal3
←   ACC0001 goal3   rss 1916 MB (+0)             ← 复用同一前向：+0MB
```

#### ② 公共网格防爆护栏（**已加，3/3 验证**）

`build_volume` 建完网格即检查体素数，超过 `GLIOMA_MAX_GRID_VOXELS`（默认 **2 亿**）→ **带完整成因分析的明确报错**：

```text
study 'ACCxxx': 公共网格 (812, 812, 812) = 535 M 体素，超过上限 200 M（GLIOMA_MAX_GRID_VOXELS）。
预计这一例的峰值内存约 36 GB —— 32GB 容器会被直接 OOM-kill。
  最常见成因：**参考序列的 spacing/affine 元数据异常**（target_grid 原样采用病态网格）。
  …（附可粘贴的 nib 排查命令、三种处置）
  容错模式（GLIOMA_LOADER_TOLERANT=1）下本例会被跳过、其余照常产出。
```

**为什么这是正确姿势**：一例的峰值 ≈ `(C+13)×V×4B`（RAM 的 volume/z-score 拷贝/EDT + GPU 的 `x0/acc/wacc/seg_prob`）。真实 1mm 脑 V≈8.9M 时峰值 <1GB；**要打爆 32GB，V 必须在 4~6 亿** —— 唯一现实成因就是病态元数据。被平台杀容器 = 整批作废且无日志；护栏报错 + 容错 = **只丢这一例**，日志里还有完整原因。三份预处理（goal5 / 共享骨干 / `glioma_goals`）**全部接入**。

```text
实测：812³=535M → 拦截 ✓   512³=134M → 放行 ✓   上限调 50M 后 64M → 拦截 ✓
```

#### ③ 两个已量化的"一次性"内存（**不是泄漏，不要追**）

| 项 | 实测 | 说明 |
|---|---|---|
| `import torch/numpy` | ~320 MB | 探针基线已扣除（`scripts/memprobe.py` 打印"基线（导入完成）"） |
| **首次前向的 lazy-init** | **+1492 MB，峰值 2830 MB** | torch CPU 的线程 workspace / oneDNN **一次性**分配。goal1 付一次，goal2/3/4 前向缓存命中后 **+1/+0 MB**。**若容器 GPU 不可见（`GLIOMA_DEVICE` 落到 cpu），这项更高且全部计入容器 RAM** —— 追踪器开头会直接打出 `CUDA 可用 : False` 警告 |

#### ④ 本轮顺带修的统计漏报

`goal2_duplicate` 的权重在 `DuplicateConfig.ckpt_rel`（`encoder.pt`），不是类属性 —— **G6 之前把它算成"无权重"**。已修，现在 G6 输出 `goal2_duplicate rel=goal2_duplicate/encoder.pt 解析出 1 份`，常驻模型总数 = **6**。

> 已知未修（记录在案）：`DuplicateTask.load_model` 不走 `_SHARED_WEIGHTS` 共享缓存（自己 `torch.load`）。它读的是独立的 `encoder.pt`，即使接入缓存也是单独的 key —— 单份 5MB 冗余，不影响正确性，暂不动（减少提交前的改动面）。

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
| `GLIOMA_MAX_ENSEMBLE` | **8** | 多折集成上限。目录里 `*.pt` 超过它 → **明确报错**而不是静默全加载（OOM 防护）。确实要多折且超过 8 折时调大 |
| `GLIOMA_MAX_GRID_VOXELS` | **200_000_000** | 公共网格体素数上限（防爆护栏）。超过 → 明确报错 + 容错跳过该例，而不是让容器被 OOM-kill（§4.10） |

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
| `test_goal5_contract.py::test_task_does_not_touch_answer_dir` 报 **`DID NOT RAISE FileNotFoundError`** | **测试依赖环境**（没钉 `COMPETITION_CHECKPOINT_ROOT`），容器里有权重 → 模型真的加载了 | 已修（§4.6）；重跑 `python -m pytest tests/contracts -q` |
| **修完空掩膜之后评测开始 OOM** | **`_bridge` 用 21³ 稠密结构元**：空掩膜时代它在 `if not m.any(): return` 就返回、从不执行；掩膜变非空后才第一次真正跑（+701 MB / 31.7 s 一次，`clean_pair` 调两次） | **已修**（§4.8）；回归测试 `test_bridge_matches_closing_and_stays_local` |
| **第一例就 OOM-kill 容器（32GB）** | 单例**瞬时峰值**而非累积泄漏；头号嫌疑：某例 spacing/affine 元数据异常 → 病态巨大网格；其次：GPU 不可见 → 全部激活落 RAM | `python3 scripts/stage_mem_trace.py 3`（日志最后一行 `→` = 元凶，§4.10）；护栏已默认拦截病态网格 |
| 跑 pytest 时容器 OOM 重启 | 测试套件实测峰值仅 372 MB，**不是元凶**；先证伪再看权重目录 / `_bridge` | `python -m pytest tests -p scripts.memprobe -q`，再 `python3 scripts/mem_audit.py --trace`（§4.7、§4.8） |
| 服务一启动 / 一评测就 OOM | 常驻模型数 = 不同 `ckpt_rel` 数 × 份数；目录里混了训练快照（`epoch_*.pt`/`last.pt`/备份） | 清目录或只留 `core.pt`；`GLIOMA_MAX_ENSEMBLE` 调上限；看预检 G6 |
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

# ---------- ② 契约测试 + 一键预检（看到 GO 再往下）----------
mkdir -p logs
python3 -m pytest tests/contracts -q 2>&1 | tee logs/contracts.log    # 期望 55 passed
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
