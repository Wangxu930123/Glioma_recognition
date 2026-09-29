# 断点判定 · 续训 · OOM 定位与解决（路线 A / `glioma_track4`）

> 适用：`CONFIG=train20 bash scripts/03_train.sh 0`（train 80% / val 20%，tag = `g4_fold0`）
> 回答三个问题：**① 现在的 `last.pth` 是第几轮的产物？② 怎么接着训？③ OOM 到底是谁吃的内存、怎么解？**
>
> 文中命令都在工程根目录执行：`cd /2026aicompetition/workspace/dcs/glioma_track4`
>
> ✅ **代码修复已落地（5 处，本次修改）**：
> ① `trainer.py`：`itertools.cycle` → `_endless`（消除 **≈9.4 GB/epoch 无界泄漏**，**P0**）；
> ② `trainer.py`：`validate()` 流式累计（消除每 epoch **9 GB** 高水位，**P1**）；
> ③ `trainer.py`：续训后 `del ck`（释放常驻的 checkpoint state_dict）；
> ④ `scripts/14_calibrate_thresholds.py`：流式累计（消除 **≈47 GB**，收尾 1/5 步）；
> ⑤ `src/evaluation/evaluate.py`：同上（消除 **≈47 GB**，收尾 2/5 步）。
> 重启训练进程即生效（Python 不会热加载）。

---

## 1. 先判断：这个 checkpoint 是第几轮的产物

### 1.1 判据只有一条：`.pth` 里的 `epoch` 字段

`trainer` 在**每个 epoch 的验证与选模之后**写盘（`src/training/trainer.py:509-513`）：

```python
torch.save({"model": ..., "model_ema": ..., "cls_spec": cls_spec,
            "epoch": epoch, "best_metric": best, "thresholds": best_thr,
            ..., "optimizer": opt.state_dict(), "config": cfg}, last_p)
```

所以：

| 文件 | 什么时候写 | 语义 |
|---|---|---|
| `last.pth` | **每个 epoch 结束**（无条件） | `epoch: N` = **第 N+1 轮已完整跑完**（含该轮 val + 选模 + 存盘） |
| `best.pth` | 仅当 `metric > best` 时 | `epoch: N` = "目前最好的一版是在第 N+1 轮选出来的" |

**⚠️ 关键语义：`epoch` 是 0 基的。** 训练循环是 `for epoch in range(start_epoch, cfg["epochs"])`，存盘存的就是这个 `epoch`（`trainer.py:401, 502, 512`）。

- `last.pth` 里 `epoch: 0` → **第 1 轮（ep0）刚跑完**，不是"还没开始"。
- 续训时 `start_epoch = epoch + 1`（`trainer.py:370`）→ 从 ep1 接着跑。

### 1.2 三条命令（任选，建议都看）

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

# ① 直接读 checkpoint（最权威）
python3 - <<'PY'
import torch, os
for name in ("last", "best"):
    p = f"checkpoints/g4_fold0/{name}.pth"
    if not os.path.exists(p):
        print(f"{name}.pth 不存在"); continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    ep = int(ck.get("epoch", -1))
    print(f"{name}.pth: epoch字段={ep} → 第 {ep+1} 轮已完成"
          f" | best_metric={ck.get('best_metric')} | thresholds={ck.get('thresholds')}"
          f" | 含optimizer={'optimizer' in ck} | 文件时间={os.path.getmtime(p):.0f}")
PY

# ② 看规范 JSONL 的进度（epoch 是 1 基，正好比 .pth 的 epoch 大 1）
tail -3 logs/g4_fold0.jsonl

# ③ 看训练日志里已打印了几个 "epoch N" 落款
grep -c '^\[g4_fold0\] epoch ' logs/train_fold0.log
```

**判读**：②里最后一行 `"epoch": k` 且 `phase: "val"` → 说明第 k 轮已跑完；与①的 `epoch字段 = k-1` 应当自洽（不自洽说明日志被截断过或跑过别的 tag）。

### 1.3 一个必须知道的坑

**`best.pth` 里没有 `optimizer`，只有 `last.pth` 有。** 所以：

- 续训**只认 `last.pth`**；
- 想"从 best 继续"只能用 `PRETRAINED=checkpoints/g4_fold0/best.pth`（`trainer.py:375-380`，只加载权重、不恢复优化器与 epoch）。

---

## 2. 断点续训：正确姿势

### 2.1 核心：**重跑同一条命令即可**（不需要任何额外参数）

`trainer.py:363-374` 的自动恢复逻辑：`last.pth` 存在 → 加载 `model` + `model_ema` + `optimizer` → `start_epoch = epoch+1` → 打印一行 `[trainer] 断点续训：epoch k/20`。

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
export CACHE_DIR=/2026aicompetition/workspace/cache

# 与上次完全相同的 CONFIG + tag（tag 由 03_train.sh 决定：TAG_PREFIX 默认 g4，折号 0 → g4_fold0）
nohup env PY=python3 PYTHON=python3 CONFIG=train20 \
      DATASET_ROOT=/2026aicompetition/datasets/training/annotation \
      CACHE_DIR=/2026aicompetition/workspace/cache \
      bash scripts/03_train.sh 0 > logs/fold0_resume_launch.log 2>&1 &
echo $! > logs/train_fold0.pid
tail -f logs/train_fold0.log          # 期望第一屏出现「断点续训：epoch N/20」
```

**续训能不能生效，只看两件事**：

1. **`checkpoints/g4_fold0/last.pth` 存在**（不存在就是从零，且不会提示）；
2. **tag 一致**：`03_train.sh` 的 tag = `${TAG_PREFIX}_fold$MODE`。改了 `TAG_PREFIX` 或折号 → 换目录 → 从零开始。

### 2.2 三个必须知道的陷阱

**陷阱 1 · 学习率会重新 warmup（最关键）**
`trainer.py:353-356` 每次都**重新构造** `OneCycleLR(total_steps=epochs×steps_per_epoch)`，恢复时只 `opt.load_state_dict`，**没有 `sched.load_state_dict`**。所以续训那一次的学习率会从 OneCycle 起点重走一遍 warmup。

> 结论：**不要用"跑 5 轮 → 停 → 续跑到 20"当调参手段**。要么一开始就把 `epochs` 设成最终值一次跑完；要么先用 `configs/train5.yaml` 单独跑一个短模型看效果。
> 因 OOM 被动中断而续训是可接受的（总比训不上好），但要意识到**质量会略打折**，且中断越频繁折扣越大。

**陷阱 2 · 改 `epochs` 会改掉 LR 曲线**
`total_steps = epochs × steps_per_epoch` 是现算的。续训时把 `epochs` 从 20 改成 30，等于换了整条 LR 曲线（而且 `last.pth` 里的 `config` 只是**元信息**，不会被用来覆盖你磁盘上的 YAML）。

**陷阱 3 · 换 `batch_size` 会同时改两件事**
`batch_size` 2→1 会让 `steps_per_epoch` **翻倍**（2662 ÷ 1 = 2662，原来是 1331），于是：
- OneCycle 的 `total_steps` 翻倍 → LR 曲线整体变化；
- **`do_global` 步数翻倍 → 每 epoch 的整脑前向翻倍** → 见 §3 的主因，**内存泄漏速率也翻倍**。

### 2.3 从零重训 / 从 best 继续（底层命令）

`03_train.sh` **不转发** `--no-resume`，所以这两件事必须直接调模块：

```bash
# 从零（无视 last.pth）
python3 -m src.training.trainer --config train20 --fold 0 --tag g4_fold0 --no-resume

# 从 best.pth 继续（只加载权重，epoch 归零重新计数）
PRETRAINED=checkpoints/g4_fold0/best.pth CONFIG=train20 bash scripts/03_train.sh 0
```

> **动 `last.pth` 之前先备份**：
> ```bash
> cp -v checkpoints/g4_fold0/last.pth checkpoints/g4_fold0/last.pth.ep$(python3 -c "import torch;print(torch.load('checkpoints/g4_fold0/last.pth',map_location='cpu',weights_only=False).get('epoch'))")
> ```

---

## 3. OOM 定位：先分清是哪一种

两种 OOM 的处置完全不同，**第一步必须分型**：

| | **主机 RAM OOM**（大概率是这个） | **GPU 显存 OOM** |
|---|---|---|
| 日志特征 | 进程**直接消失**、日志最后一行戛然而止、出现 `Killed` | `torch.cuda.OutOfMemoryError: CUDA out of memory` |
| 退出码 | `137`（= 128+9 SIGKILL） | 非 0，但进程是自己抛异常退出的 |
| `nvidia-smi` | 显存**正常/不涨**，是系统在杀进程 | 显存**打满** |
| 核查命令 | `dmesg -T \| grep -iE "killed process\|oom-kill"` | `grep -i "out of memory" logs/train_fold0.log` |

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4

# 分型①：是不是被内核杀的
dmesg -T 2>/dev/null | grep -iE "killed process|out of memory|oom-kill" | tail -5

# 分型②：是不是 CUDA OOM
grep -iE "CUDA out of memory|OutOfMemoryError" logs/train_fold0.log | tail -3

# 分型③：看日志尾部是不是"戛然而止"
tail -5 logs/train_fold0.log

# 分型④：边跑边盯内存（这是最直接的证据）
watch -n 30 'free -g; echo ---; ps -o pid,rss,etime,cmd -p "$(cat logs/train_fold0.pid)" 2>/dev/null'
```

---

## 4. 根因（按证据强度排序）

### P0 · `itertools.cycle` 无界缓存 —— **主机 RAM 的主因，且是代码缺陷**

```395:396:glioma_track4/src/training/trainer.py
    sp_iter = itertools.cycle(aux["special"]) if aux.get("special") else None
    pr_iter = itertools.cycle(aux["pair"]) if aux.get("pair") else None
```

`itertools.cycle` 的语义是：**首次遍历时把每一个元素都存进内部列表**，之后靠重放这个列表来"无限循环"。也就是说，**每 `next()` 一次，就永久多占一份 batch 的内存，永不释放。**

再看这个 DataLoader 有多长（`trainer.py:261-264`）：

```python
ds_sp = SpecialImageDataset(cases, pos_fake, pos_comp, pre_cfg=pre, aug_cfg=cfg,
                            seed=cfg["seed"] + _seed_off(fold), n_per_epoch=10 ** 6)
aux["special"] = DataLoader(ds_sp, batch_size=n_spec, shuffle=False,
                            num_workers=0, drop_last=True)
```

`SpecialImageDataset.__len__` 返回 `max(self.n_per_epoch, 1)` = **10⁶**（`src/data/dataset.py:951-952`），
于是这个 DataLoader 的 `len` = 10⁶ ÷ 2 = **500 000 个 batch**。首轮遍历根本不可能跑完 —— 而 `cycle` 就在这 50 万个 batch 上**一路往列表里塞**。

**量化（这才是 OOM 的时间表）**：

| 量 | 值 | 依据 |
|---|---|---|
| 一个整脑视图 | `4×96³×4B` = **14.16 MB** | `configs/train.yaml → global_view.out: 96` + 4 通道 |
| 一个 special batch（`special_batch: 2`） | **28.3 MB** | `aux.special_batch: 2` |
| 每个 epoch 的 `do_global` 次数 | 1331 ÷ 4 ≈ **333** | `steps/epoch=1331`、`global_view.every: 4` |
| **special 每 epoch 只增不减** | 333 × 28.3 MB ≈ **9.4 GB** | ← **OOM 主因** |
| pair 首次遍历一次性缓存 | 2 项 ×(14.16+14.16) MB = 56.6 MB × `len(gold)×4÷2` 个 batch | `pair_batch: 2`；随后不再增长 |

`pair_batch` 那条（`DuplicatePairDataset.__len__` 有限）会**封顶**；`special` 这条**不封顶**。

> **`last.pth` 只到 epoch 0，却已经 OOM，说明**：进程启动基线（Python + torch + CUDA context ≈ 3~5 GB）+ worker 缓存（P1，1~3 GB）+ `pair` 封顶（2~5 GB）+ `special` 累计。也就是说，**若 `epoch` 字段到 3~6 才死，几乎可以断定就是 P0**（9.4 GB × N 的线性增长）。
> 顺带解释一个反直觉现象：**把 `batch_size` 降到 1 会让它死得更快**（`steps/epoch` 翻倍 → 每 epoch 泄漏翻倍，见 §2.2 陷阱 3）。

### P1 · `validate()` 每个 epoch 一次性收集全部 val 概率体（**约 9 GB 高水位**）

```283:290:glioma_track4/src/training/trainer.py（修复前）
    probs, gts = [], []
    for batch in dl:
        ...
        probs.append(torch.sigmoid(out["seg"].float())[0].cpu().numpy())
        gts.append(batch["target"][0].numpy())
```

旧实现把**整份 val 的预测概率体与真值体全留在内存里**，最后才逐阈值扫 Dice：

| 量 | 值 |
|---|---|
| 单例概率体（`2×96³ float32`） | 6.75 MB |
| 单例真值体（`2×96³ float32`） | 6.75 MB |
| val = 666 例 | 666 × 13.5 MB ≈ **9.0 GB** |

而且 `val_every: 1` → **每个 epoch 都出现一次这个高水位**。它本身是"有界但极大"（不是无界泄漏），
但 P0 已经把 RSS 抬高到临界点，这一下就会把进程顶穿 —— 所以它和 P0 必须一起修。

> 已改为**流式累计**：对每个 (通道, 阈值) 维护两个浮点累加器，逐例算完即丢弃体积。
> 常驻内存 `2 × 18 × 2 个浮点数`（约几百字节），**与 val 例数无关**。
> 已用随机数据验证：新旧的阈值选择与 Dice **逐位完全相同**（浮点累加次序未变）。

### P2 · DataLoader worker 的整脑视图 / 体积 LRU（主机 RAM，常数项）

`GliomaDataset._whole` 的 LRU 上限是 `max(16, cache_size×4)` = 16 个整脑视图（`src/data/dataset.py:820`），
**而且每个 worker 各持一份**（`configs/train.yaml → num_workers: 4`）：

- 16 × 14.16 MB = 227 MB / worker → 4 workers ≈ **0.9 GB**（README §16 按更宽的估计记 ~1.8 GB）；
- 另外 `GliomaDataset._cache` 会缓存 `(vol, tgt, aff)`，**未命中 mmap 快路径**时 4 项 × ~170 MB ≈ 0.7 GB/worker（命中缓存则不发生）。

### P3 · GPU 侧：一个 step 最多 4 次前向，`pair` 那一次样本数翻倍

```415:459:glioma_track4/src/training/trainer.py
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                out = model(x)                       # (a) patch 96³, batch 2
                l_seg, p_seg = seg_total_loss(out, y, cfg)
            l_seg.backward()
            ...
            if do_global:
                gout = model(batch["whole"]...)       # (b) 整脑 96³, batch 2
            ...
            if sp_iter is not None and do_global:
                gs = model(sb["image"]...)            # (c) 特殊影像 batch 2
            ...
            if pr_iter is not None and do_global:
                pair_x = torch.cat([pb["a"]..., pb["b"]...], dim=0)
                gp = model(pair_x)                     # (d) **4 个样本一次过网**
```

每段都是"前向 → backward → 释放"，峰值大致等于**最重那一次的前向**，而 (d) 的样本数是 batch_size 的 2 倍。
`global_view.out=96` 与训练 patch `96³` 同尺寸，所以 (b)(c)(d) 的激活量与 (a) 同量级。

### P4 · 两个工程抢同一台机器

路线 B（`glioma_goals` 六个 goal）里 `val` 是**逐例前向**且各 goal 有自己的 cache；路线 A 又同时开 4 个 DataLoader worker。
**这是唯一会真正"互相破坏"的路径**：任一进程把主机内存/显存吃满，另一个会被拖死或被杀。

---

## 5. 解决方案

### 方案 A（**根治**）· 已应用到 `src/training/trainer.py`

把 `itertools.cycle` 换成不缓存的生成器 `_endless`（已随本次修改落地，并顺手删掉了不再使用的 `import itertools`）：

```python
def _endless(loader):
    """无限迭代一个 DataLoader，但不缓存历史 batch（cycle 会把首次遍历的每个元素常驻内存）。"""
    while True:
        for batch in loader:
            yield batch
```

```python
# 用 _endless 而不是 itertools.cycle：后者会把每个 batch 永久缓存（见 _endless 注释）
sp_iter = _endless(aux["special"]) if aux.get("special") else None
pr_iter = _endless(aux["pair"]) if aux.get("pair") else None
```

**效果**：每 epoch 的常驻内存回到**常数**；`special`/`pair` 的监督信号强度**完全不变**
（仍是每 `do_global` 步取一个 batch，轮完一轮自动从头再来，语义与 `cycle` 一致）。
**生效条件**：改完必须**停掉并重启**训练进程（Python 不会热加载）。

### 方案 A2（**根治**）· 已应用到 `validate()`

把"先收集全量概率体再扫阈值"改成**流式累计**（每通道每阈值两个累加器）：

```python
stats = [[[0.0, 0.0] for _ in grid] for _ in range(2)]     # [通道][阈值] = [分子, 分母]
for batch in dl:
    ...
    p = torch.sigmoid(out["seg"].float())[0].cpu().numpy()   # 只保留当前这一例
    g = batch["target"][0].numpy()
    for c in range(2):
        gt = g[c] > 0.5
        gsum = float(gt.sum())
        for i, t in enumerate(grid):
            pred = p[c] > t
            stats[c][i][0] += 2.0 * float((pred & gt).sum())
            stats[c][i][1] += float(pred.sum()) + gsum
```

**效果**：常驻内存从 O(val 例数)（666 例 ≈ 9 GB）降到 O(阈值数)（几百字节）。
**等价性**：已验证新旧的**阈值选择与 Dice 逐位完全相同**（对任一累加器而言，参与相加的用例顺序没变）。

### 方案 A3（卫生项）· 已应用

续训 / 预训练加载完 checkpoint 后立即 `del ck`（它含 `model`/`model_ema`/`optimizer` 三份
state_dict），避免整份权重被函数作用域一直引用到训练结束。

### 方案 B（不改代码的减压组合，只能延缓、不能根治）

| 手段 | 做法 | 效果 / 代价 |
|---|---|---|
| 低内存配置 | `CONFIG=train20_lowmem`（`batch_size` 2→1、`num_workers` 4→1） | worker 侧 -0.7 GB；**但 P0 泄漏速率翻倍**（§2.2 陷阱 3）→ 见下方 ⚠️ |
| 降整脑前向频率 | `global_view.every: 4 → 8` | 泄漏速率**减半**（4.7 GB/epoch）；代价：目标①②④ 学得更慢 |
| 减少 worker | `num_workers: 1`（只改这一项，别动 batch） | P1 从 ~0.9 GB 降到 ~0.23 GB；代价：数据吞吐下降 |
| 分段跑 | 跑 N 轮 → 停 → 重跑同命令续训 | 内存归零重来；**代价是 OneCycle 重新 warmup**（§2.2 陷阱 1） |

> ⚠️ **`train20_lowmem` 单独用是危险的**：它把 `batch_size` 降到 1，`steps/epoch` 从 1331 变 2662，
> `do_global` 次数翻倍 → P0 的泄漏从 9.4 GB/epoch 变成 **18.9 GB/epoch**。
> 所以 **`train20_lowmem` 必须和"方案 A 的补丁"或"`every: 8` + 分段跑"一起用**，否则会更早死。
> 若只是想省显存、不想动 batch，优先改 `num_workers` 和 `global_view.every`。

### 方案 C（GPU 显存侧的旋钮，按性价比排序）

1. `global_view.every` 4 → 8（同时省显存与时间）；
2. `num_workers` 4 → 2（省的是主机内存与 CPU，几乎不影响显存）；
3. `batch_size` 2 → 1（**注意 P0 的副作用**，见上）；
4. **不要**优先降 `patch_size`：`preprocess.yaml → inference.patch` 是 96³，训练 patch 必须与推理一致，不一致会显著掉点；
5. **单卡不要设 `CUDA_VISIBLE_DEVICES=1`**（会把唯一的卡隐藏掉，`torch.cuda.is_available()` 变 False）。

### 方案 A4（**根治**）· 已应用到收尾评估的两个脚本

**这一处比 A2 严重一个数量级**，必须单独说明：

`scripts/14_calibrate_thresholds.py`（`16_finalize.sh` 的 **1/5 步**）与
`src/evaluation/evaluate.py`（**2/5 步**）用的是和 `validate()` 修复前同型的写法，
**但它们收集的是全图体积，不是 patch**：

```163:166:glioma_track4/src/inference/sliding.py
    seg_roi = (acc / wacc.clamp_min(1e-6))[0, :, :d, :h, :w].float()
    seg_prob = torch.zeros((2, vol.shape[1], vol.shape[2], vol.shape[3]), device=dev)
    seg_prob[:, off[0]:off[0] + d, off[1]:off[1] + h, off[2]:off[2] + w] = seg_roi
    seg_prob = seg_prob.cpu().numpy()
```

即 `res["seg"]` 被**贴回原尺寸** `(2, D, H, W)`；`make_targets(...)` 同尺寸。

| 量 | 值 | 依据 |
|---|---|---|
| 单例 full-volume `vol`（4 通道） | 71 MB | `dataset.py:649` 注释"避免每例全量载入 71MB" |
| 单例 `res["seg"]` | ≈35.5 MB | `2 × D×H×W × 4B` |
| 单例 `gt` | ≈35.5 MB | 同上 |
| **单例 probs + gts** | **≈71 MB** | |
| `14_calibrate_thresholds.py`（`--limit 0` = 全 666 例） | **≈47 GB** | 32 GB 容器**死在约 380 例处** |
| `evaluate.py`（同样全 666 例） | **≈47 GB** | 同上 |

> 结论：**32 GB 容器下，`16_finalize.sh` 的 1/5 与 2/5 步都会必然 OOM** ——
> 训练就算跑完了，收尾也拿不到指标。所以这两处和 A/A2 一样是**必修项**。

改法与 A2 完全一致（流式累计 → O(1) 常驻），并已验证**与旧结果逐位等价**
（直接用真实模块 `src.evaluation.evaluate` 的 `_accumulate` / `_best_from_stats`
对同一批随机数据跑，旧 `_thr_scan` 与新实现的阈值与 Dice 完全相同）。
`_thr_scan` 已随之移除（全仓仅此一处引用）。

### 推荐顺序

```
① 分型（§3）→ CUDA OOM 走 C；主机 RAM OOM（大概率）走 ②
② 方案 A / A2 / A3 —— **本次已全部应用**，无需再改
③ 重启续训（同 tag，自动从 last.pth 的 epoch+1 开始）
④ 若还想更稳：只把 num_workers 降到 2（不要动 batch_size）
⑤ 全过程用 watch free -g 盯住 RSS 是否变成"围绕一个水位小幅波动"
```

---

## 6. 可直接粘贴的执行序列

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
export DATASET_ROOT=/2026aicompetition/datasets/training/annotation
export CACHE_DIR=/2026aicompetition/workspace/cache
export WORKSPACE=/2026aicompetition/workspace

########## ① 先确认现在停在哪一轮 ##########
python3 - <<'PY'
import torch, os
p = "checkpoints/g4_fold0/last.pth"
print("last.pth 存在:", os.path.exists(p))
if os.path.exists(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    ep = int(ck.get("epoch", -1))
    print(f"epoch字段={ep} → 第 {ep+1} 轮已完成；best_metric={ck.get('best_metric')}")
    print("续训将从 epoch", ep + 1, "开始")
PY
tail -2 logs/g4_fold0.jsonl

########## ② 备份 last.pth（改代码/换配置前必备份）##########
cp -v checkpoints/g4_fold0/last.pth checkpoints/g4_fold0/last.pth.bak

########## ③ 分型 OOM ##########
dmesg -T 2>/dev/null | grep -iE "killed process|oom-kill" | tail -3
grep -iE "CUDA out of memory" logs/train_fold0.log | tail -2

########## ④ 打「方案 A」补丁（见 §5，改 src/training/trainer.py 共 6 行）##########

########## ⑤ 续训（tag 不变 → 自动读 last.pth）##########
nohup env PY=python3 PYTHON=python3 CONFIG=train20 \
      DATASET_ROOT=/2026aicompetition/datasets/training/annotation \
      CACHE_DIR=/2026aicompetition/workspace/cache \
      WORKSPACE=/2026aicompetition/workspace \
      bash scripts/03_train.sh 0 > logs/fold0_resume_launch.log 2>&1 &
echo $! > logs/train_fold0.pid
sleep 30 && head -20 logs/train_fold0.log      # 必须看到「断点续训：epoch N/20」

########## ⑥ 盯内存（关键：RSS 应"围绕水位波动"，不再单调爬升）##########
watch -n 30 'free -g; echo ---; grep VmRSS /proc/$(cat logs/train_fold0.pid)/status 2>/dev/null'

########## ⑦ 训完后收尾（不要在训练中途跑）##########
SKIP_MOCK=1 FOLDS="0" bash scripts/16_finalize.sh
bash scripts/09_export_submission.sh --verify
```

---

## 7. 验证清单

**断点判定**

- [ ] `last.pth` 的 `epoch` 字段 = 已完成的轮数 - 1（0 基）
- [ ] `logs/g4_fold0.jsonl` 最后一行 `epoch` 比它大 1，且 `phase` 是 `val`（自洽）
- [ ] 备份文件 `last.pth.bak` 已生成

**续训**

- [ ] 启动日志出现 `[trainer] 断点续训：epoch N/20`（没有这行 = 没续上，检查 tag 与路径）
- [ ] `logs/train_fold0.log` 的第一个 `epoch` 行号 = N（不是 0）
- [ ] 清楚知道续训那一次 LR 会重新 warmup（这就是不要"跑几轮停一停"的原因）

**OOM**

- [ ] 已分型：主机 RAM（`Killed`/137）还是 GPU（`CUDA out of memory`）
- [ ] 代码侧已确认：`grep -n "itertools.cycle" src/training/trainer.py` **只出现在注释里**
- [ ] 代码侧已确认：`grep -n "def _endless" src/training/trainer.py` 有定义，且两处 `cycle` 已被替换
- [ ] 修复后 RSS **不再随 epoch 单调爬升**（每 epoch 不再 +9.4 GB）
- [ ] 验证期不再出现 ~9 GB 的瞬时尖峰（`validate` 已流式化）
- [ ] `batch_size` 与 `num_workers` 的改动是**有意识**的（知道降 batch 会加速 P0 泄漏）
- [ ] 若共卡跑路线 B：确认两个进程的峰值 RAM 之和 < 容器内存上限

---

## 8. 一句话小结

- **判断**：`torch.load(...)["epoch"]` 是 **0 基**，`epoch: 0` = 第 1 轮已跑完。
- **续训**：**重跑同一条命令**（同 `CONFIG`、同 tag），自动从 `last.pth` 的 `epoch+1` 开始；代价是 OneCycle 会重新 warmup。
- **OOM 主因（已修）**：`trainer.py` 原来的 `itertools.cycle` 会**永久缓存每个 batch**，而 special 的 DataLoader 长度是 50 万 → **约 9.4 GB/epoch 的单调泄漏**，与 epoch 数线性相关。
  实测证实：迭代 200 次（每 batch 1 MB）后，`itertools.cycle` 常驻 **200.1 MB**，`_endless` 常驻 **1.0 MB**（200×）。
- **OOM 帮凶一（已修）**：`validate()` 每个 epoch 把全部 val 概率体+真值体留在内存 → 666 例 ≈ **9 GB 瞬时高水位**。
- **OOM 帮凶二（已修，比前者大一个数量级）**：收尾的 `14_calibrate_thresholds.py` / `evaluate.py`
  收集的是**全图体积**（≈71 MB/例）→ 666 例 ≈ **47 GB**，32 GB 容器**必然在收尾阶段 OOM**。
- **解决**：`itertools.cycle` → `_endless`（方案 A）、`validate` 流式化（A2）、`del ck`（A3）、
  两个评估脚本流式化（A4）。
  `train20_lowmem` 之类的降配**只能延缓、且降 `batch_size` 会加速泄漏**（steps/epoch 翻倍）。
