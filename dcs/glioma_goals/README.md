# 赛道四 · 五目标独立训练工程

> 用途：**五个人并行训练五个任务，互不影响**。
> 每个 Goal 是一个自包含的训练工程：配置、代码、缓存、权重、日志全部在自己的目录内。

---

## 1. 目录结构

```text
glioma_goals/
├── shared/                      # 公共库（1962 行）——**只读，请勿修改**
│   ├── backbone_mednext.py      # 主骨干（MedNeXt 风格）
│   ├── backbone_unet3d.py       # 备选骨干（ResUNet）
│   ├── selector.py              # 序列选择（模态识别）
│   ├── volume.py                # 多通道体积预处理（1mm 公共网格）
│   ├── spatial.py               # 世界坐标重采样与逆变换
│   ├── sliding.py               # 滑窗 + TTA 推理
│   ├── data.py                  # 数据集基类 / patch / 增强 / 掩码发现
│   ├── losses.py                # 通用损失组件
│   ├── engine.py                # 训练循环（EMA / AMP / warmup 余弦）
│   ├── metrics.py               # 指标
│   ├── factory.py               # 按 checkpoint 元信息重建网络
│   └── utils.py                 # 配置 / 日志 / 路径
│
├── goal1_authenticity/          # ① 影像真实性（假人体 / 非人体）
├── goal2_stitched/              # ②-A 拼接影像检测
├── goal2_duplicate/             # ②-B 重复影像检测（配对任务）
├── goal3_tumor/                 # ③ 病灶识别（胶质瘤 vs 非肿瘤）
├── goal4_diagnosis/             # ④ 辅助诊断（14 个结构化字段）
└── goal5_segmentation/          # ⑤ 分割（T1C 核心区 + FLAIR 周边区）
```

每个 Goal 目录内：

| 文件 | 作用 | 可改？ |
|---|---|---|
| `config.yaml` | 超参、数据路径、损失权重 | ✅ 随便改 |
| `train.py` | 训练入口 | ✅ |
| `dataset.py` | 本任务的标签构造 | ✅ |
| `losses.py` | 本任务的损失与指标 | ✅ |
| `model.py` | 网络（默认引用 `shared` 骨干） | ✅ |
| `evaluate.py` | 独立评估 | ✅ |
| `README.md` | 本任务说明与已知限制 | ✅ |
| `runs/` `cache/` | 训练产物（运行时自动创建） | 不要提交 |

## 2. 并行训练（五个人同时跑，互不干扰）

每个人进自己的目录、占一张卡：

```bash
# 你 —— 目标五
cd goal5_segmentation && CUDA_VISIBLE_DEVICES=0 python train.py --tag my_exp

# 同事A —— 目标一
cd goal1_authenticity && CUDA_VISIBLE_DEVICES=1 python train.py --tag alice

# 同事B —— 目标二B
cd goal2_duplicate && CUDA_VISIBLE_DEVICES=2 python train.py --tag bob

# 同事C —— 目标三
cd goal3_tumor && CUDA_VISIBLE_DEVICES=3 python train.py --tag carol
```

**为什么不会互相干扰**（三条隔离保证）：

1. **写入隔离**：每个 Goal 的权重写在 `runs/<tag>/checkpoints/`、
   日志写在 `runs/<tag>/logs/`，没有任何共享写路径；
2. **缓存隔离**：预处理缓存在各自的 `cache/`。共享缓存看似省磁盘，
   但两个进程同时写会产生半截文件，且损坏后不报错、只让指标莫名变差；
3. **验证集隔离**：每个 Goal 按自己的 `train.val_ratio` 与 `seed` 划分验证集，
   不需要等别人先产出统一的折划分文件。

唯一共享的是**只读**的 `shared/` 与数据集本身。

## 3. 开始训练

```bash
cd <goal>/
python train.py --limit 40        # ① 先冒烟（几分钟，确认数据/依赖就绪）
python train.py                   # ② 正式训练
python evaluate.py --ckpt runs/<goal>/checkpoints/best.pth
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--tag` | 实验名，产物落在 `runs/<tag>/`（多个实验互不覆盖） |
| `--data` | 数据根目录（默认取 `config.yaml` 的 `data.root`） |
| `--limit` | 只用前 N 个病例，用于冒烟 |
| `--epochs` / `--batch-size` / `--lr` | 覆盖配置 |
| `--device` | `cuda`（默认）或 `cpu` |

## 4. 数据约定

```text
<data_root>/
└── <AccessionNumber>/
    ├── <SeriesUid>/xxx.nii.gz          # 影像（不得含掩码关键词）
    └── <SeriesUid>/<掩码>.nii.gz        # 掩码（瘤体/水肿/肿瘤瘤体/mask/seg…）
```

- 影像统一归一到 **1mm 公共网格**，通道顺序固定为 `t1c, flair, t2, t1`；
- 缺模态**零占位**（不会崩，但会记为 warning）；
- **掩码角色由"文件名 + 所在序列的模态"共同判定**：
  FLAIR/T2 上的"瘤体"属于**周围总异常区(peri)**，不是核心区(core)；
  同一角色的多个掩码（如 FLAIR 下的"瘤体"+"水肿"）取**并集**。
- 结构化字段（目标四）从病例目录下的 `label.json` 读取；
  缺失字段会被 mask 掉，不参与损失。

## 5. 已知限制（务必如实记录，不要粉饰）

| 目标 | 限制 |
|---|---|
| ① 真实性 | 正样本极少（本地仅 6 例），AUC 置信区间很宽，泛化能力未知 |
| ②-A 拼接 | 同上（6 例） |
| ②-B 重复 | 金标准对极少（12 对），主要靠自监督增强对撑起训练 |
| ③ 病灶识别 | **本地数据全是胶质瘤，无非肿瘤负样本 → AUC 在数学上无法评估**，当前用"有无掩码"占位。需补脑梗死/脑脓肿病例 |
| ④ 辅助诊断 | 依赖 `label.json`；数据缺标注时损失会被跳过（已加显式告警） |
| ⑤ 分割 | 相对最成熟；但仍需全图评估确认（训练时看到的是 patch 级 Dice） |

## 6. 排查清单

| 现象 | 可能原因 |
|---|---|
| `dice` 恒为 0 | 掩码没被识别（命名不含关键词）或掩码与影像空间不一致 |
| 目标四 `loss=0` 且不变 | 病例缺 `label.json`（看日志里的 ⚠️ 告警） |
| 多个 Goal 训练互相拖慢 | 是否共用同一张 GPU（各自指定 `CUDA_VISIBLE_DEVICES`） |
| val 指标为 `nan` | 验证集里没有正样本（正样本本来就极少，属于已知限制） |

## 7. 与提交工程的关系

本工程是**研发侧**：每人训练自己那一路，产出独立权重。
比赛提交侧（`Glioma_recognition-main`）负责把权重接到统一 Pipeline 上，
两者的网络结构通过 `shared/factory.build_shared_backbone` 保持同源，
因此**训练出来的权重可以直接被提交工程加载**。
