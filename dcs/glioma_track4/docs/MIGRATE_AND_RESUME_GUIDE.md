# 代码搬迁 + 断点续训 完全手册

> 适用：把工程换到另一个目录（或换回原目录）之后，**接着原来那一轮训练继续跑**。
> 场景：`glioma_track4` 路线 A（`CONFIG=train20`，tag = `g4_fold0`，折内 8:2）。
>
> 关联文档：
> - [`OOM_AND_RESUME_GUIDE.md`](OOM_AND_RESUME_GUIDE.md) —— OOM 根因、修复与内存监控
> - [`ROUTE_A_TRAIN_README.md`](ROUTE_A_TRAIN_README.md) —— 本方案（折内 8:2、20 epoch）的完整流程
>
> ✅ **本手册的所有命令都可整段粘贴**；每段开头都自带变量定义，互不依赖。

---

## 0. 三十秒速览：先对号入座

| 场景 | 你的状态 | 看哪一节 | 关键动作 |
|---|---|---|---|
| **C（推荐）** | 新代码在 `.../GliomaRecognition/dcs/`，想**换回原位置** `.../workspace/dcs/` | [§3](#3-场景-c推荐代码换回原位置--拷回权重续训) | 先备份 → 改名旧目录 → 搬新代码进来 → 拷回 `checkpoints/`+`data/` |
| **B** | 就想在**新目录**里继续跑，不动原目录 | [§4](#4-场景-b在新目录直接续训) | 把 `checkpoints/` + `data/` 从旧目录搬过去 |
| **A** | 代码位置没变，只是想接着跑 | [§5](#5-场景-a原地续训最简) | **重跑同一条命令**即可 |

**三条铁律**（违反任一条都会静默出问题，不会报错）：

1. **先备份，再动手** —— 备份必须放在"即将被删/被改名的目录"**之外**。
2. **用 `mv` 改名代替 `rm`** —— 留着回滚路，跑稳了再删。
3. **tag 必须还是 `g4_fold0`** —— 换 tag = 换产物目录 = 从零开始。

---

## 1. 先搞清：哪些东西在 git 里，哪些不在

**这一节决定了你「必须搬什么」。** 工程用 `.gitignore` 明确排除了所有训练产物：

```17:19:glioma_track4/.gitignore
/checkpoints/
*.pth
*.pt
```
```31:33:glioma_track4/.gitignore
/data/*
/logs/
*.jsonl
```

所以 **`git clone` / `git pull` 拿到的新目录里只有代码，没有权重、没有数据清单、没有日志**。

| 类别 | 具体项 | 位置 | 搬不搬 |
|---|---|---|---|
| **必须搬** | `checkpoints/g4_fold0/last.pth` | 工程内 | ✅ **续训的唯一依据**（含 `epoch`/`best_metric`/`optimizer`） |
| **必须搬** | `checkpoints/g4_fold0/best.pth` | 工程内 | ✅ 最优权重，收尾/导出用 |
| **必须搬** | `data/folds.json` | 工程内 | ✅⚠️ **不带就会重建划分 → val 换另一批病例 → 已训权重与训练集不对应（评估泄漏，且不报错）** |
| **必须搬** | `data/manifest.json` | 工程内 | ✅ 数据清单；重建要重跑探针，且病例顺序一变划分也跟着变 |
| 建议搬 | `logs/train_fold0.log`、`logs/g4_fold0.jsonl` | 工程内 | 🔸 只为日志连续（JSONL 是 append，规范上更完整） |
| **绝不搬** | `data/manifest_val.json` | 工程内 | ❌ 它的存在会让评估切到**无掩膜的官方验证集**，还会在它上面标定阈值写回权重 |
| **不用搬** | `CACHE_DIR`（预处理缓存） | `/2026aicompetition/workspace/cache` | ❌ 绝对路径、在私有存储，按 accession 存 → **继续命中**（重建很贵，**别删**） |
| **不用搬** | 数据根 | `/2026aicompetition/datasets/training/annotation` | ❌ 平台挂载点，不在工程目录里 |
| **不用搬** | 提交权重 / 答案 / 规范日志 | `/2026aicompetition/workspace/{checkpoint,answer,logs}` | ❌ 绝对路径，与工程目录无关 |

**千万别手滑删掉**：`/2026aicompetition/datasets/`（数据挂载点，动了全盘皆输）、
`/2026aicompetition/workspace/{cache,checkpoint,logs,answer}`。

---

## 2. 动手前必做：确认新代码里有没有该有的修复

`*.pth`、`data/`、`logs/` 被 gitignore —— **但代码改动不是**。
如果你本地改过源码却没 `commit` + `push`，新拉下来的代码里就**没有这些修复**，
典型后果是**收尾阶段（`16_finalize.sh`）OOM**。

```bash
# 把路径改成你的"新代码"位置
NEW=/2026aicompetition/workspace/dcs/GliomaRecognition/dcs/glioma_track4
cd "$NEW"

echo -n "trainer._endless（应为 3）        : "; grep -c "_endless" src/training/trainer.py
echo -n "evaluate._best_from_stats（应≥3） : "; grep -c "_best_from_stats" src/evaluation/evaluate.py
echo -n "阈值标定 n_used（应≥2）           : "; grep -c "n_used" scripts/14_calibrate_thresholds.py
echo -n "残留的 itertools.cycle 代码行     : "
grep -n "itertools.cycle" src/training/trainer.py | grep -v -E ':\s*(#|")' | wc -l
```

| 输出 | 判定 |
|---|---|
| `3 / ≥3 / ≥2 / 0` | ✅ 修复齐全，可以继续 |
| 任一为 `0` | ❌ 新代码缺该修复 → 按 §3.4 从旧目录（或你本地）覆盖这几个文件 |

> 💡 **根治办法**：把你本地这几个文件的改动 `git add` + `commit` + `push`，以后无论怎么拉都不会丢。

```bash
git add glioma_track4/src/training/trainer.py \
        glioma_track4/src/evaluation/evaluate.py \
        glioma_track4/scripts/14_calibrate_thresholds.py \
        glioma_track4/src/data/dataset.py \
        glioma_track4/configs/train20_lowmem.yaml \
        glioma_track4/docs/OOM_AND_RESUME_GUIDE.md \
        glioma_track4/docs/MIGRATE_AND_RESUME_GUIDE.md
git commit -m "fix(mem): 消除 cycle 无界泄漏 + 评估/标定流式化；补搬迁续训手册"
git push
```

---

## 3. 场景 C（推荐）：代码换回原位置 + 拷回权重续训

**为什么推荐这个**：回到原位置后，`paths.yaml` 的绝对路径、三个工程的同级关系、
`{WORKSPACE}/checkpoint/` 导出路径**全都和从前一致**，一行命令都不用改。

### 3.0 变量（每段命令都自带，可重复执行）

```bash
DCS=/2026aicompetition/workspace/dcs                      # 工程父目录（三个工程并列于此）
NEW=$DCS/GliomaRecognition/dcs                            # ★ 新代码当前所在
STAMP=20260929                                            # 备份/改名后缀，随便改
KEEP=/2026aicompetition/workspace/_resume_keep_$STAMP      # ★ 保命备份（在 DCS 之外）
TRK=$DCS/glioma_track4
```

### 3.1 停掉正在跑的进程（有的话必须先停）

两个目录 tag 都是 `g4_fold0`，会变成两个独立进程争同一张卡 / 同一份内存；
且两者导出到**同一个绝对路径** `{WORKSPACE}/checkpoint/`，**谁后导出谁覆盖**。

```bash
DCS=/2026aicompetition/workspace/dcs
pgrep -af "src.training.trainer"                                    # 先看清有哪些
kill "$(cat $DCS/glioma_track4/logs/train_fold0.pid)" 2>/dev/null || true
sleep 3
pgrep -af "src.training.trainer" || echo "已无训练进程 ✓"
```

### 3.2 备份（★ 此刻旧目录还没被动过，这是唯一的安全窗口）

```bash
DCS=/2026aicompetition/workspace/dcs
STAMP=20260929
KEEP=/2026aicompetition/workspace/_resume_keep_$STAMP
TRK=$DCS/glioma_track4

mkdir -p "$KEEP/checkpoints/g4_fold0" "$KEEP/data" "$KEEP/logs"
rsync -av "$TRK/checkpoints/g4_fold0/" "$KEEP/checkpoints/g4_fold0/"
cp -v  "$TRK/data/manifest.json" "$TRK/data/folds.json" "$KEEP/data/"
rsync -av "$TRK/logs/" "$KEEP/logs/" 2>/dev/null || true

# 记下指纹，第 3.6 步要逐字节校验
md5sum "$KEEP/data/folds.json" "$KEEP/data/manifest.json" "$KEEP/checkpoints/g4_fold0/last.pth" \
  | tee "$KEEP/checksums.txt"

# ★ 闸门：这三个必须在，否则不许往下走
ls -l "$KEEP/checkpoints/g4_fold0/last.pth" "$KEEP/data/folds.json" "$KEEP/data/manifest.json"
```

### 3.3 顺便把"旧目录里的代码"也留一份（回滚 + 补修复都用它）

```bash
DCS=/2026aicompetition/workspace/dcs
STAMP=20260929
KEEP=/2026aicompetition/workspace/_resume_keep_$STAMP
for d in glioma_track4 glioma_goals Glioma_recognition-main; do
  [ -d "$DCS/$d" ] && rsync -a --exclude checkpoints --exclude logs "$DCS/$d/" "$KEEP/code_old/$d/"
done
ls "$KEEP/code_old/"        # 期望看到三个工程目录
```

### 3.4 查修复 → 缺就补

先按 [§2](#2-动手前必做确认新代码里有没有该有的修复) 检查 `$NEW`。
**若发现有缺失**，用旧目录（或你本地已推送的分支）里的版本覆盖：

```bash
DCS=/2026aicompetition/workspace/dcs
NEW=$DCS/GliomaRecognition/dcs
OLD=$DCS/glioma_track4                     # 旧目录（第 3.5 步才会改名，这里还是原名）

for f in src/training/trainer.py src/evaluation/evaluate.py scripts/14_calibrate_thresholds.py; do
  if ! diff -q "$OLD/$f" "$NEW/glioma_track4/$f" >/dev/null 2>&1; then
    echo "→ 覆盖缺失/不一致：$f"
    cp -v "$OLD/$f" "$NEW/glioma_track4/$f"
  fi
done
# 再复跑一次 §2 的检查，四项都要符合期望
```

### 3.5 替换：旧目录**改名**（不是删），新代码搬到原位置

```bash
DCS=/2026aicompetition/workspace/dcs
NEW=$DCS/GliomaRecognition/dcs
STAMP=20260929

for d in glioma_track4 glioma_goals Glioma_recognition-main; do
  [ -d "$DCS/$d" ] && mv "$DCS/$d" "$DCS/${d}.old-$STAMP"
done

mv "$NEW/glioma_track4" "$NEW/glioma_goals" "$NEW/Glioma_recognition-main" "$DCS/"

echo "--- 当前 DCS 下 ---"
ls -d "$DCS"/*/
# ★ 闸门：必须同时看到 glioma_track4 / glioma_goals / Glioma_recognition-main 三者并列
#   （脚本里有写死的相对关系，缺一个后面会失败）
```

### 3.6 拷回保命文件 + 自检

```bash
DCS=/2026aicompetition/workspace/dcs
STAMP=20260929
KEEP=/2026aicompetition/workspace/_resume_keep_$STAMP
TRK=$DCS/glioma_track4

mkdir -p "$TRK/checkpoints/g4_fold0" "$TRK/data" "$TRK/logs"
rsync -av "$KEEP/checkpoints/g4_fold0/" "$TRK/checkpoints/g4_fold0/"
cp -v  "$KEEP/data/manifest.json" "$KEEP/data/folds.json" "$TRK/data/"
rsync -av "$KEEP/logs/" "$TRK/logs/" 2>/dev/null || true

# ⚠️ 必须清掉验证集清单：它的存在会让收尾切到无掩膜的官方验证集
rm -f "$TRK/data/manifest_val.json"

cd "$TRK"
echo "=== ① 指纹校验（三个文件都要 OK）==="
md5sum -c "$KEEP/checksums.txt"

echo "=== ② 权重可读 + 知道从第几轮接 ==="
python3 - <<'PY'
import torch
ck = torch.load("checkpoints/g4_fold0/last.pth", map_location="cpu", weights_only=False)
need = ["model", "model_ema", "optimizer", "epoch", "best_metric", "thresholds"]
print("缺失键  :", [k for k in need if k not in ck] or "无 ✓")
ep = int(ck.get("epoch", -1))
print(f"epoch字段 = {ep}  →  第 {ep+1} 轮已完成，将从 epoch {ep+1} 续训")
print("best_metric =", ck.get("best_metric"), "| thresholds =", ck.get("thresholds"))
PY

echo "=== ③ 修复检查（四项都要符合期望）==="
echo -n "_endless=";             grep -c "_endless" src/training/trainer.py
echo -n "_best_from_stats=";     grep -c "_best_from_stats" src/evaluation/evaluate.py
echo -n "n_used=";               grep -c "n_used" scripts/14_calibrate_thresholds.py

echo "=== ④ 划分口径 ==="
python3 -c "import json;f=json.load(open('data/folds.json'));print('fold0: train',len(f['0']['train']),'/ val',len(f['0']['val']))"
# 期望：fold0: train 2662 / val 666
```

### 3.7 启动续训

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
mkdir -p logs

nohup env PY=python3 PYTHON=python3 CONFIG=train20 \
      DATASET_ROOT=/2026aicompetition/datasets/training/annotation \
      CACHE_DIR=/2026aicompetition/workspace/cache \
      WORKSPACE=/2026aicompetition/workspace \
      bash scripts/03_train.sh 0 > logs/fold0_resume_launch.log 2>&1 &
echo $! > logs/train_fold0.pid

sleep 30
echo "=== 启动检查（两行都必须出现）==="
grep -E "断点续训|tag=g4_fold0" logs/train_fold0.log
```

必须看到：

```text
[trainer] 断点续训：epoch N/20
[trainer] tag=g4_fold0 fold=0 train=2662 val=666
```

| 检查 | 通过 | 不通过怎么办 |
|---|---|---|
| `断点续训：epoch N/20` | ✅ 权重读到了 | 没有这行 = **静默从零开始**。查 §3.6 ①（`last.pth` 是否在 `checkpoints/g4_fold0/`）、tag 是否被 `TAG_PREFIX` 改过 |
| `train=2662 val=666` | ✅ 划分没被换掉 | 数字不对 = `folds.json` 没拷对，**立刻停车**重做 §3.6 |

### 3.8 跑稳之后再清理（别急）

```bash
DCS=/2026aicompetition/workspace/dcs
STAMP=20260929
rm -rf "$DCS"/*.old-$STAMP                                # 旧代码
rmdir "$DCS/GliomaRecognition/dcs" "$DCS/GliomaRecognition" 2>/dev/null   # 嵌套空壳
rm -rf /2026aicompetition/workspace/_resume_keep_$STAMP    # 备份（确认一切正常后再删）
```

---

## 4. 场景 B：在新目录直接续训

不想动原目录时用这条。**注意**：新目录里 `paths.yaml` 用的是绝对路径 → 行为与原地一致；
但三个工程必须仍然**同级**（`ls -d .../dcs/*/` 应看到三者）。

```bash
OLD=/2026aicompetition/workspace/dcs/glioma_track4
NEW=/2026aicompetition/workspace/dcs/GliomaRecognition/dcs/glioma_track4

# ① 从旧目录搬保命文件
mkdir -p "$NEW/checkpoints/g4_fold0" "$NEW/data" "$NEW/logs"
rsync -av "$OLD/checkpoints/g4_fold0/" "$NEW/checkpoints/g4_fold0/"
cp -v  "$OLD/data/manifest.json" "$OLD/data/folds.json" "$NEW/data/"
rsync -av "$OLD/logs/" "$NEW/logs/" 2>/dev/null || true
rm -f "$NEW/data/manifest_val.json"          # 别把它带过来

# ② 停掉旧目录的进程（否则两个进程争同一张卡、导出还会互相覆盖）
kill "$(cat $OLD/logs/train_fold0.pid)" 2>/dev/null || pkill -f "src.training.trainer"

# ③ 续训
cd "$NEW" && mkdir -p logs
nohup env PY=python3 PYTHON=python3 CONFIG=train20 \
      DATASET_ROOT=/2026aicompetition/datasets/training/annotation \
      CACHE_DIR=/2026aicompetition/workspace/cache \
      bash scripts/03_train.sh 0 > logs/fold0_resume_launch.log 2>&1 &
echo $! > logs/train_fold0.pid
sleep 30; grep -E "断点续训|tag=g4_fold0" logs/train_fold0.log
```

---

## 5. 场景 A：原地续训（最简）

代码位置没变时，**什么都不用做，重跑同一条命令**。`trainer.py` 会自动读 `last.pth`：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
mkdir -p logs
nohup env PY=python3 PYTHON=python3 CONFIG=train20 \
      DATASET_ROOT=/2026aicompetition/datasets/training/annotation \
      CACHE_DIR=/2026aicompetition/workspace/cache \
      bash scripts/03_train.sh 0 > logs/fold0_resume_launch.log 2>&1 &
echo $! > logs/train_fold0.pid
sleep 30; grep -E "断点续训|tag=g4_fold0" logs/train_fold0.log
```

---

## 6. 一键脚本（整段粘贴，场景 C）

带闸门：任一步失败就 `exit`，不会带着半截状态往下跑。

```bash
#!/usr/bin/env bash
set -euo pipefail

DCS=/2026aicompetition/workspace/dcs
NEW=$DCS/GliomaRecognition/dcs
STAMP=20260929
KEEP=/2026aicompetition/workspace/_resume_keep_$STAMP
TRK=$DCS/glioma_track4

echo "########## 0) 停进程 ##########"
pkill -f "src.training.trainer" 2>/dev/null || true
sleep 3

echo "########## 1) 备份（唯一安全窗口）##########"
mkdir -p "$KEEP/checkpoints/g4_fold0" "$KEEP/data" "$KEEP/logs"
rsync -a "$TRK/checkpoints/g4_fold0/" "$KEEP/checkpoints/g4_fold0/"
cp "$TRK/data/manifest.json" "$TRK/data/folds.json" "$KEEP/data/"
rsync -a "$TRK/logs/" "$KEEP/logs/" 2>/dev/null || true
cd "$KEEP" && md5sum data/folds.json data/manifest.json checkpoints/g4_fold0/last.pth > checksums.txt
test -f "$KEEP/checkpoints/g4_fold0/last.pth" || { echo "✗ 备份里没有 last.pth"; exit 1; }
for d in glioma_track4 glioma_goals Glioma_recognition-main; do
  rsync -a --exclude checkpoints --exclude logs "$DCS/$d/" "$KEEP/code_old/$d/" 2>/dev/null || true
done
echo "备份完成 → $KEEP"

echo "########## 2) 补缺失的修复 ##########"
for f in src/training/trainer.py src/evaluation/evaluate.py scripts/14_calibrate_thresholds.py; do
  diff -q "$TRK/$f" "$NEW/glioma_track4/$f" >/dev/null 2>&1 || cp -v "$TRK/$f" "$NEW/glioma_track4/$f"
done

echo "########## 3) 替换（旧目录改名，不删）##########"
for d in glioma_track4 glioma_goals Glioma_recognition-main; do
  [ -d "$DCS/$d" ] && mv "$DCS/$d" "$DCS/${d}.old-$STAMP"
done
mv "$NEW/glioma_track4" "$NEW/glioma_goals" "$NEW/Glioma_recognition-main" "$DCS/"
ls -d "$DCS"/*/

echo "########## 4) 拷回 + 清掉验证集清单 ##########"
mkdir -p "$TRK/checkpoints/g4_fold0" "$TRK/data" "$TRK/logs"
rsync -a "$KEEP/checkpoints/g4_fold0/" "$TRK/checkpoints/g4_fold0/"
cp "$KEEP/data/manifest.json" "$KEEP/data/folds.json" "$TRK/data/"
rsync -a "$KEEP/logs/" "$TRK/logs/" 2>/dev/null || true
rm -f "$TRK/data/manifest_val.json"

echo "########## 5) 自检闸门 ##########"
cd "$TRK"
md5sum -c "$KEEP/checksums.txt"
python3 - <<'PY'
import torch, json
ck = torch.load("checkpoints/g4_fold0/last.pth", map_location="cpu", weights_only=False)
need = ["model", "model_ema", "optimizer", "epoch", "best_metric", "thresholds"]
miss = [k for k in need if k not in ck]
assert not miss, f"缺失键: {miss}"
ep = int(ck.get("epoch", -1))
f = json.load(open("data/folds.json"))
print(f"✓ 将从 epoch {ep+1} 续训 | fold0: train {len(f['0']['train'])} / val {len(f['0']['val'])}")
assert len(f["0"]["val"]) > 0, "folds.json 内容异常"
PY
grep -q "_endless" src/training/trainer.py || { echo "✗ trainer.py 缺 _endless 修复"; exit 1; }
grep -q "_best_from_stats" src/evaluation/evaluate.py || { echo "✗ evaluate.py 缺流式修复"; exit 1; }
grep -q "n_used" scripts/14_calibrate_thresholds.py || { echo "✗ 标定脚本缺流式修复"; exit 1; }
echo "自检全部通过 ✓"

echo "########## 6) 启动续训 ##########"
mkdir -p logs
nohup env PY=python3 PYTHON=python3 CONFIG=train20 \
      DATASET_ROOT=/2026aicompetition/datasets/training/annotation \
      CACHE_DIR=/2026aicompetition/workspace/cache \
      WORKSPACE=/2026aicompetition/workspace \
      bash scripts/03_train.sh 0 > logs/fold0_resume_launch.log 2>&1 &
echo $! > logs/train_fold0.pid
sleep 30
grep -E "断点续训|tag=g4_fold0" logs/train_fold0.log \
  || { echo "✗ 没看到续训行，检查日志：logs/fold0_resume_launch.log"; exit 1; }
echo "✅ 已在后台续训；监控：tail -f logs/train_fold0.log"
```

---

## 7. 验收清单

**搬迁**

- [ ] 备份在 `$DCS` **之外**（`/2026aicompetition/workspace/_resume_keep_*`）
- [ ] 备份里有 `last.pth` / `best.pth` / `manifest.json` / `folds.json` 四件
- [ ] 旧目录是**改名**（`.old-<STAMP>`）而不是删除
- [ ] `$DCS` 下三个工程**并列**：`glioma_track4` / `glioma_goals` / `Glioma_recognition-main`
- [ ] `md5sum -c` 三个文件全 `OK`
- [ ] `data/manifest_val.json` **不存在**（已 `rm` 或改名）

**续训**

- [ ] 启动日志有 `[trainer] 断点续训：epoch N/20`
- [ ] 启动日志有 `tag=g4_fold0 fold=0 train=2662 val=666`（数字必须一致）
- [ ] `logs/train_fold0.log` 里第一个 `epoch` 行号 = N（不是 0）
- [ ] 清楚知道**续训那一次 LR 会重新 warmup**（`OneCycleLR` 不恢复调度器，见 `OOM_AND_RESUME_GUIDE.md` §2.2）

**代码**

- [ ] `_endless` / `_best_from_stats` / `n_used` 三项检查全过
- [ ] 这三个修复已 `git commit` + `push`（下次拉取不会再丢）

**跑稳之后**

- [ ] 跑满 ≥1 个 epoch，`RSS 横盘`（内存不随 epoch 爬升）
- [ ] 之后才删 `.old-*` 与 `_resume_keep_*`

---

## 8. 常见故障对照

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| ① | 启动日志**没有**「断点续训」 | `checkpoints/g4_fold0/last.pth` 不在（没搬 / 搬错目）；或 tag 被 `TAG_PREFIX` 改了 | §3.6 ① 核对；确认命令里没有 `TAG_PREFIX=...` |
| ② | 有「断点续训」但 `train/val` 数字不对 | `folds.json` 不是原来那份（被重建或拿错） | 立刻停车，重做 §3.2 + §3.6；**不要**让训练继续（划分已不对应） |
| ③ | 启动后从 `epoch 0` 开始 | 同 ①：没读到 `last.pth` | 同上 |
| ④ | `03_train.sh: command not found` / 相对路径报错 | 三个工程不在同一层 | §3.5 的闸门：`ls -d "$DCS"/*/` 必须三个并列 |
| ⑤ | 收尾 `16_finalize.sh` 被杀（`Killed`） | 新代码缺流式修复（§2 检查为 0） | §3.4 覆盖那三个文件后重跑收尾 |
| ⑥ | 评估走成了 external 口径 / 阈值被带偏 | `data/manifest_val.json` 又出现了 | `rm -f data/manifest_val.json`；`FOLDS="0" bash scripts/16_finalize.sh --print-split` 应显示 `split=oof` |
| ⑦ | 两个进程同时跑、结果互相覆盖 | 旧目录的进程没停 | §3.1 先 `pkill`，再启动新的 |
| ⑧ | 显存/内存被挤爆 | 共卡跑着路线 B | 只保留一个训练进程；或按 `OOM_AND_RESUME_GUIDE.md` §5 降配 |
| ⑨ | 预处理很慢、缓存像没生效 | `CACHE_DIR` 没设或目录被删 | `ls /2026aicompetition/workspace/cache \| head`；`paths.yaml` 默认已指向它，别删 |

---

## 9. 附录 · 命令速查

| 目的 | 命令 |
|---|---|
| 看还有谁在跑 | `pgrep -af "src.training.trainer"` |
| 看停在第几轮 | `python3 -c "import torch;c=torch.load('checkpoints/g4_fold0/last.pth',map_location='cpu',weights_only=False);print(c['epoch']+1,'轮已完成')"` |
| 看折划分规模 | `python3 -c "import json;f=json.load(open('data/folds.json'));print('train',len(f['0']['train']),'val',len(f['0']['val']))"` |
| 看 external 口径是否被误开 | `FOLDS="0" bash scripts/16_finalize.sh --print-split`（期望 `split=oof`） |
| 原地续训 | `CONFIG=train20 bash scripts/03_train.sh 0` |
| 从零重训 | `python3 -m src.training.trainer --config train20 --fold 0 --tag g4_fold0 --no-resume` |
| 从 best 继续 | `PRETRAINED=checkpoints/g4_fold0/best.pth CONFIG=train20 bash scripts/03_train.sh 0` |
| 监控内存 | 见 `OOM_AND_RESUME_GUIDE.md` §4（`logs/mem_watch.sh`） |
| 收尾 | `SKIP_MOCK=1 FOLDS="0" bash scripts/16_finalize.sh` |
| 导出 + 复核 | `bash scripts/09_export_submission.sh 0` → `bash scripts/09_export_submission.sh --verify` |

---

## 10. 一句话小结

**git 里只有代码**（`/checkpoints/`、`*.pth`、`/data/*`、`/logs/` 都被 ignore），
所以换目录时必须亲手搬 **`checkpoints/g4_fold0/` 整个目录 + `data/{folds.json,manifest.json}`**；
`folds.json` 尤其关键 —— 不带它，折划分会被重建，已训权重就和它的训练集不对应了。
搬完用 **`断点续训：epoch N/20`** 和 **`train=2662 val=666`** 两行日志确认，
再照 `OOM_AND_RESUME_GUIDE.md` §4 盯住 RSS 是否横盘。
