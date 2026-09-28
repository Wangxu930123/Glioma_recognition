# 路线 A · 一体化多任务训练 README

> 工程：`glioma_track4`　|　入口：`scripts/03_train.sh`
>
> 本文只讲**训练**。评估/导出/提交见 `CLOUD_USAGE_GUIDE.md`（工作区根目录）；
> 三工程总体说明见 `MASTER_GUIDE.md`。

---

## 0. 路线 A 是什么

**一个多任务骨干，一次前向同时产出六路输出**，每个 optimization step 六个目标一起前进：

| 损失项（`configs/train.yaml` → `loss`） | 对应目标 |
|---|---|
| `dice` + `ce` + `deep_sup` | ⑤ 分割（core / peri 两通道 + 深监督） |
| `cls` | ③ 病灶识别（`TumorProbability`）+ ④ 14 个结构化字段 |
| `special` | ① 影像真实性（假人体）+ ②-A 拼接 |
| `embed` | ②-B 重复影像（配对嵌入） |
| `contain` | 集合约束 core ⊆ peri |

因此**不存在"goal1 训完再训 goal2"** —— 六个目标天然同步推进，一个 checkpoint 覆盖全部六个
（导出时同一份权重会被复制到六个 `checkpoint/<goal>/` 目录）。

与**路线 B（`glioma_goals` 六个独立单任务工程）**的区别：

| | 路线 A（本文） | 路线 B |
|---|---|---|
| 组织 | 1 个多任务模型 | 6 个单任务模型 |
| 折支持 | 折训练 + **全量训练**都有 | 只有折/`val_ratio` |
| 断点续训 | ✅ 默认开启 | ❌ 无 |
| 阈值标定/评估/导出 | `16_finalize.sh` 一条龙 | 需自己跑 `evaluate.py` |
| 提交推荐 | ✅（推理插件依赖多任务头） | ⚠️ 需实测 |

---

## 1. 前置条件

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
export VAL_ROOT=/2026aicompetition/datasets/verification
export CACHE_DIR=/2026aicompetition/workspace/cache      # 缓存放私有存储，容器删了不丢
export WORKSPACE=/2026aicompetition/workspace            # 日志自动落到 $WORKSPACE/logs
```

两张清单必须先就位：

| 清单 | 生成命令 | 谁需要 |
|---|---|---|
| `data/manifest.json`（训练集） | `bash scripts/01_probe.sh` | 所有模式 |
| `data/manifest_val.json`（官方验证集） | `bash scripts/01_probe.sh --val` | **只有 `full` 模式需要** |

> `paths.yaml` 里 `raw.val` 已按平台路径填好，不 export `VAL_ROOT` 也能跑；
> `DATASET_ROOT` 同理（默认 `.../datasets/training`，会自动下钻到 `annotation/`）。

**全量模式的硬前提**：验证集里必须有**带掩膜**的病例。全量模式的 val 只用带掩膜的病例算 Dice
来选 best，一例都没有时训练器会直接退出（`src/training/trainer.py:203-217`）。先自检：

```bash
python3 - <<'PY'
import json
m = json.load(open("data/manifest_val.json"))
print("验证集病例:", len(m["cases"]), "| 带掩膜:", sum(1 for c in m["cases"] if c.get("masks")))
PY
```

可选但强烈建议（把重采样开销提前做掉，训练时直接命中缓存）：

```bash
python3 scripts/13_build_cache.py --workers 8
```

---

## 2. 三种训练模式

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

# ① 单折（前台，产物 checkpoints/g4_fold0/）
bash scripts/03_train.sh 0
#    指定折号：bash scripts/03_train.sh 2    /   FOLD=2 bash scripts/03_train.sh

# ② 多折并行（4 张卡 → GPU0..GPU3；第 5 折后补 bash scripts/03_train.sh 4）
bash scripts/03_train.sh all 4

# ③ 全量训练（train=清单全部病例，val=官方验证集，无折划分）
bash scripts/03_train.sh full
```

| 模式 | tag（→ `checkpoints/<tag>/`） | 日志 |
|---|---|---|
| 单折 | `g4_fold<N>` | `logs/train_fold<N>.log` |
| 多折并行 | `g4_fold0..` | `logs/train_fold<N>.log`（各折一份） |
| 全量 | `g4_full` | `logs/train_full.log` |

可用的环境变量：

| 变量 | 默认 | 作用 |
|---|---|---|
| `CONFIG` | `train` | 用哪份 `configs/*.yaml`（如 `CONFIG=train20`） |
| `TAG_PREFIX` | `g4` | 产物目录前缀（冒烟用 `TAG_PREFIX=smoke` → `checkpoints/smoke_full/`） |
| `PRETRAINED` | 空 | 加载预训练权重（`PRETRAINED=/path/x.pth`，合规性自行确认） |
| `FOLD` | `$1` | 单折模式的折号 |
| `PY` | 自动探测 | 解释器（容器里常只有 `python3`） |

> ⚠️ `03_train.sh` **不转发** `--no-resume`。需要"无视断点、从零重训"时直接调底层入口：
> `python3 -m src.training.trainer --config train20 --fold full --tag g4_full --no-resume`

---

## 3. 全量训练 20 epoch（本方案）

`epochs` 写在**配置文件**里（`03_train.sh` 只透传 `--config`，没有 `--epochs` 参数）。
工程已备好 `configs/train20.yaml`（唯一差异：`epochs: 20`）：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

# 后台跑（推荐，SSH 断了不死）
CONFIG=train20 nohup bash scripts/03_train.sh full > logs/full20_launch.log 2>&1 &
tail -f logs/train_full.log
```

启动成功会看到这两行：

```text
[trainer] tag=g4_full fold=full train=<全部病例> val=<带掩膜验证例数> steps/epoch=... 
[trainer] 全量模式：train=xxxx（清单全部病例） val=xxx（官方验证集 .../manifest_val.json）
```

想只训 5 个 epoch 快速看效果（OneCycle 会在 5 轮内完整退火，属于"正经训完"的短模型）：

```bash
sed 's/^epochs: 20/epochs: 5/' configs/train20.yaml > configs/train5.yaml
CONFIG=train5 bash scripts/03_train.sh full
```

---

## 4. 配置旋钮（`configs/train20.yaml`）

| 键 | 默认 | 说明 / 什么时候改 |
|---|---|---|
| `epochs` | 20 | 总轮数。**OneCycle 的学习率退火跨度 = 这里**，所以要设成你真正打算跑完的轮数 |
| `batch_size` | 2 | 显存不够降到 1 |
| `patch_size` | `[96,96,96]` | 显存不够降到 `[80,80,80]`（82³ 也在文档允许范围） |
| `num_workers` | 4 | CPU 核多可到 8 |
| `val_every` | 1 | 每 N 轮验证一次。**建议保持 1**（见 §6 陷阱 2） |
| `lr` / `weight_decay` | 3e-4 / 3e-5 | `OneCycleLR(max_lr=lr, total_steps=epochs×steps_per_epoch)` |
| `grad_clip` | 12.0 | 早期梯度爆炸时调小 |
| `amp_dtype` | `bfloat16` | 平台 CUDA12.8 支持；老卡改成 `float16` |
| `checkpoint_metric` | `val_dice_peri` | 选 `best.pth` 的依据；也可 `val_dice_mean` |
| `model.*` | mednext / base 32 / depth 4 | `arch: resunet` 可回退轻量版；`depth: 5` + `max_ch` 换大模型 |
| `global_view.every` | 4 | **每 N 个 step 做一次整脑全局前向**（分类/特殊/嵌入头）。调大省时间、调小更准 |
| `global_view.size_mm / out` | 192 / 96 | **必须与推理侧一致**（`preprocess.yaml → inference.global_size`），别改 |
| `aux.special_batch` | 2 | 每步目标①②样本数；为 0 则该头不训练 |
| `aux.pair_batch` | 2 | 每步重复影像配对数；为 0 则嵌入头不训练 |
| `loss.*` | 见文件 | 六路权重；某个目标明显弱就调大对应项 |
| `folds` / `val_ratio` | 5 / 0.2 | **只有折训练用**，`full` 模式忽略 |

---

## 5. 训练中怎么读日志

**每个 step（每 `log_every=20` 步一行）** —— 能同时看到六路分项在降：

```text
[g4_full] ep0 step20/1331 loss=1.2345 {'seg': 0.8123, 'cls': 0.3012, 'special': 0.4442, 'embed': 0.0221}
```

**每个 epoch 一行** —— 这是选模指标：

```text
[g4_full] epoch 0 loss=0.9876 dice_core=0.6123 dice_peri=0.7011 thr=[0.4, 0.5] (1832s)
```

| 看见什么 | 含义 |
|---|---|
| `loss` 单调下降、`dice_*` 上升 | 正常 |
| `special ≈ 0.69` 不动 | 该头没有监督信号（`aux.special_batch=0` 或正样本为空），看 `[loss][告警]` |
| `dice_peri` 长期 ≈ 0 | 掩膜没被识别 / 验证集没有掩膜 / 输入通道全零（看启动时的"没有真输入通道"提示） |
| `[trainer] 断点续训：epoch 3/20` | 上次的 `last.pth` 被复用（正常行为，不是重跑） |

两份日志：

| 文件 | 内容 |
|---|---|
| `logs/train_full.log` | 终端 stdout 的 tee（人看的） |
| `$WORKSPACE/logs/g4_full.jsonl` | 规范要求的 JSONL（`timestamp/epoch/step/phase/loss/lr/mode/data_source/checkpoint`，**epoch 从 1 开始**） |

---

## 6. 断点续训与两个必须知道的陷阱

**续训默认开启。** `src/training/trainer.py:363-374`：只要 `checkpoints/<tag>/last.pth` 存在，
就会加载 `model` + `model_ema` + `optimizer`，并从 `epoch+1` 继续。重跑同一条命令即可。

**陷阱 1 —— 续训时学习率会重爬。**
`OneCycleLR` 每次运行都**重新构造**（代码只 `opt.load_state_dict`，没有 `sched.load_state_dict`），
所以续训那一次的学习率会从 OneCycle 起点再走一遍 warmup。

> 结论：**别用"跑 5 轮 → 停 → 续跑到 20"的方式**。要么一开始就把 `epochs` 设成最终值一次跑完，
> 要么先用 `epochs: 5` 单独跑一个短模型看效果。

**陷阱 2 —— `val_every > 1` 会写出假的 `best.pth`。**
未验证的 epoch 里 `vm` 是全零字典，`metric = dice_peri = 0.0`；而 `best` 初值是 `-1.0`，
`0.0 > -1.0` 成立 → 会保存一个 **dice=0 的 best.pth**。保持 `val_every: 1` 就不会撞上。

**中途停掉**：`Ctrl+C` 或 `kill`。最多丢掉**当前这一轮**的进度，上一轮的 `last.pth` 还在，
下次重跑自动从那里续。所以停训练是安全的。

---

## 7. 产物位置

```text
glioma_track4/
├── checkpoints/<tag>/
│   ├── best.pth     ← 按 checkpoint_metric 选优；**导出提交用这个**
│   └── last.pth     ← 每个 epoch 都写（含 optimizer）；断点续训读这个
└── logs/
    ├── train_full.log / train_fold<N>.log
    └── ...
$WORKSPACE/logs/<tag>.jsonl      ← 规范要求的 JSONL 日志
```

`best.pth` 里带的元信息（推理侧依赖）：`model` / `model_ema` / `cls_spec` / `arch` / `model_cfg` /
`global_size` / `global_size_mm` / `thresholds` / `epoch` / `best_metric` / `special_trained` / `config`。

---

## 8. 训练完做什么

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

# 只想看指标（不导出）
CKPT=checkpoints/g4_full/best.pth bash scripts/04_eval.sh --split external

# 收尾：阈值标定 + 官方验证集评估 + 导出提交权重（+ Mock 闭环）
SKIP_MOCK=1 FOLDS=full bash scripts/16_finalize.sh
bash scripts/09_export_submission.sh --verify
```

完整提交流程（自检、本地推理、Mock 协议、镜像）见 `CLOUD_USAGE_GUIDE.md` §9-§10。

---

## 9. 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `[trainer] 全量训练（--fold full）需要官方验证集 ... 还没有验证集清单` | 缺 `data/manifest_val.json` | `export VAL_ROOT=... && bash scripts/01_probe.sh --val` |
| `... 验证集清单里**没有一例带掩膜**（Dice 算不出来，选不了 best）` | 官方验证集无掩膜 | 用折训练 `bash scripts/03_train.sh 0`；或让数据方补掩膜 |
| `ValueError: 数据根 ... 指向数据集父目录` | `DATASET_ROOT` 填到了 `/2026aicompetition/datasets` | 指到 `.../training`（预期行为，不是 bug） |
| 扫到 0 例 / `series_type_rows = 0` | 数据根偏了一层，或没读到 `SeriesType.xlsx` | 先 `python3 scripts/29_locate_dataset_root.py`；见 `CLOUD_USAGE_GUIDE.md` §4 |
| `CUDA out of memory` | patch/batch 太大，或与别的训练共卡 | 降 `patch_size` / `batch_size` / `val_every`；共卡时留足显存 |
| 单卡上设 `CUDA_VISIBLE_DEVICES=1` 后 `torch.cuda.is_available()` 变 False | 单卡机器只有 0 号卡 | 留空（默认 0）或显式 `=0` |
| 训练日志一行不出、卡在 `01_probe` / 取数阶段 | 数据没挂载或表没接上 | 看 `CLOUD_USAGE_GUIDE.md` §4 的三个判据 |

---

## 10. 一句话小结

```bash
# 折训练（要 OOF / 多折集成）
bash scripts/03_train.sh 0

# 全量训练（不要交叉验证，train=全部病例，val=官方验证集）
bash scripts/01_probe.sh --val
CONFIG=train20 bash scripts/03_train.sh full
SKIP_MOCK=1 FOLDS=full bash scripts/16_finalize.sh
```
