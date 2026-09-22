# 目标二-B · 重复影像检测

> 负责人：**待分配**　|　工程目录：`goal2_duplicate`

跨检查配对，输出 `duplicate_pairs.jsonl` 的概率

## 快速开始

```bash
cd goal2_duplicate
python train.py --limit 40          # ① 先冒烟：确认数据/依赖就绪（几分钟）
python train.py                     # ② 正式训练（产物在 runs/goal2_duplicate/）
python evaluate.py --ckpt runs/goal2_duplicate/checkpoints/best.pth
```

## 与其他 Goal **并行训练**（互不影响）

本目录是**完全独立**的训练工程：配置、数据缓存、权重、日志全部落在自己目录内。

```bash
# 五个人同时训练，各占一张卡，互不干扰：
CUDA_VISIBLE_DEVICES=0 python train.py     # 你（本目录）
CUDA_VISIBLE_DEVICES=1 python train.py     # 同事在 goal2_stitched/
CUDA_VISIBLE_DEVICES=2 python train.py     # 同事在 goal3_tumor/
```

唯一共享的是**只读**的 `../shared/` 与数据集本身：

- `../shared/` 是公共库（骨干/数据管线/增强/训练引擎/指标），
  **请不要修改**——改它会影响所有人的实验可比性。需要定制就在本目录里
  覆盖同名文件（`dataset.py` / `losses.py` 本来就是你的独立副本）。
- 数据集只读；缓存写在 `goal2_duplicate/cache/`，各 Goal 物理隔离。

## 本目录文件

| 文件 | 作用 | 可改？ |
|---|---|---|
| `config.yaml` | 超参、数据路径、损失权重 | ✅ 随便改 |
| `train.py` | 训练入口 | ✅ |
| `dataset.py` | 本任务的标签构造（继承 `shared.data.BaseCaseDataset`） | ✅ |
| `losses.py` | 本任务的损失与指标 | ✅ |
| `model.py` | 网络（默认引用 `shared` 骨干） | ✅ |
| `evaluate.py` | 独立评估 | ✅ |
| `runs/` `cache/` | 训练产物（自动创建） | — 不要提交 |

## 设计要点

- **1mm 公共网格**：不同序列层厚常不一致（T1C 1mm / FLAIR 3mm），
  不统一网格则多通道无法对齐，掩膜也无法写回原始空间；
- **逐通道 z-score**：MRI 强度无绝对物理意义，跨序列全局归一化会破坏对比；
- **几何增强同时作用于影像与掩码**，且掩码用最近邻重采样
  （线性插值会产生 0.5 这类中间值，污染二值监督信号）；
- **缺模态零占位**并保持通道顺序固定（`t1c, flair, t2, t1`），语义稳定。

## 已知限制

无（如有，请在此记录，便于复盘与对外说明）。
