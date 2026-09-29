# 用已保存模型做推理与测评 + 提交测试结果 完全手册

> 目的：**趁训练还在跑，先用已经落盘的 `best.pth` 把「推理 → 测评 → 提交」这条链路整条走一遍，专找 bug。**
> 走通了再等训练结束、做收尾、发起正式测评。
>
> 关联文档：
> - [`OOM_AND_RESUME_GUIDE.md`](OOM_AND_RESUME_GUIDE.md) —— 内存/断点续训
> - [`MIGRATE_AND_RESUME_GUIDE.md`](MIGRATE_AND_RESUME_GUIDE.md) —— 代码搬迁与权重真伪判定
> - [`PLATFORM_GUIDE.md`](PLATFORM_GUIDE.md) —— 云桌面 / 训推平台零经验版（提交那一步的界面路径看它）
>
> 所有命令都能整段粘贴；每段自带变量定义，互不依赖。

---

## 0. 三十秒速览

```text
① 打模型快照（必须先做！训练每轮都在重写 best.pth，torch.save 非原子）
② A. 本地算指标（快，支持 --limit）        → 字段 / 分割
③ B. 本地跑提交工程（慢，放后台）           → 导出 → 契约测试 → local_eval → 合规校验 → 协议复现
④ C. 提交到平台（网页操作）                → 测评容器 → 验证测评 → 初赛测评
```

**三条铁律**

1. **先打快照再评估** —— 别直接读 `checkpoints/g4_fold0/best.pth`（训练正在写它）。
2. **现在绝对不要跑 `16_finalize.sh`** —— 它的 1/5 步 `14_calibrate_thresholds.py --write` 会**写回阈值、改 `best.pth`**，与训练抢同一个文件。收尾必须等训练结束。
3. **不设 `COMPETITION_PIPELINE_FACTORY` = 静默跑 Dummy 基线** —— 服务正常、答案格式合规，但**没有真实模型输出**。

---

## 1. 前提：给模型打一个"能读通过"的快照

训练**每个 epoch 都重写** `checkpoints/g4_fold0/best.pth`，而 `torch.save` **不是原子写**（先截断再写）。
评估进程若在写入瞬间去读，会拿到半个文件 → `invalid load key` / `PytorchStreamReader failed`。

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
SNAP=/2026aicompetition/workspace/_snap
mkdir -p "$SNAP"

for i in 1 2 3 4 5; do
  cp -f checkpoints/g4_fold0/best.pth "$SNAP/best.pth"
  if python3 -c "import torch,sys;ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False);assert 'model' in ck;print('[snap] OK  epoch=',ck.get('epoch'),' best_metric=',ck.get('best_metric'),' thresholds=',ck.get('thresholds'))" "$SNAP/best.pth" 2>/dev/null; then
    break
  fi
  echo "[snap] 第 $i 次读到不完整文件（训练正在写）→ 5 秒后重试"; sleep 5
done
ls -l "$SNAP/best.pth"
```

同时把**训练进度**记下来（后面判断"用哪份权重"要用）：

```bash
grep -E "断点续训|从零训练|tag=g4_fold0" logs/train_fold0.log | head -3
grep -c '^\[g4_fold0\] epoch ' logs/train_fold0.log       # 已跑完几轮
```

> ⚠️ 快照里的 `epoch` 决定了这是"第几轮的模型"。**刚跑 2~3 轮的权重指标一定很差**（`dice_peri ≈ 0.7` 是爬升中段），
> 所以 **B 段的价值在"链路有没有 bug"，不在"分数多少"**。分数等收尾后再看。

---

## 2. A 段 · 本地算指标（快，先做这些）

用**折内 val**（666 例，**有掩膜也有字段金标准**）。官方验证集没有金标准，算不了指标。

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
export PY=python3 PYTHON=python3
export CACHE_DIR=/2026aicompetition/workspace/cache
SNAP=/2026aicompetition/workspace/_snap/best.pth

# ① 字段（目标③④）：20 例自检 → 全量 666 例
python3 scripts/35_eval_fields.py --split fold --fold 0 --ckpt "$SNAP" --limit 20
python3 scripts/35_eval_fields.py --split fold --fold 0 --ckpt "$SNAP"

# ② 分割（Dice / NSD / HD95）：20 例 → 全量
CKPT="$SNAP" bash scripts/04_eval.sh --split fold 0 20
CKPT="$SNAP" bash scripts/04_eval.sh --split fold 0
```

| 脚本 | 产出 | 看哪里 |
|---|---|---|
| `35_eval_fields.py` | 14 个字段 accuracy / precision / recall / F1 / AUC | stdout 表格 + `logs/eval_fields_fold0.json` |
| `04_eval.sh` → `evaluate.py` | Dice / NSD / HD95 + 重复影像 | `logs/eval_fold0.log` |

⚠️ **目标①②（假人体 / 拼接 AUC）现在测不了**：`15_eval_special_dup.py --split oof` 会**自动按折读 `checkpoints/g4_fold<f>/best.pth`，不接受 `--ckpt` 快照** → 会直接读正在被写的文件。等训练结束后再跑：

```bash
python3 scripts/15_eval_special_dup.py --split oof        # 训练结束后再跑
```

### A 段要顺便确认的"没有 bug"

- [ ] `35_eval_fields.py` 能跑完不报错，表格里 14 个字段都出现了
- [ ] `TumorProbability` 行若显示 `AUC = nan` + "无负样本" → **这是数据决定的**（本地没有非肿瘤性病变），不是 bug
- [ ] `04_eval.sh` 的 `logs/eval_fold0.log` 里有 `n_no_mask`（跳过数），且 Dice 不是 `nan`
- [ ] 两者都**没有**走 external 口径（`--split fold` 时不应出现 "外部验证集"）

---

## 3. B 段 · 本地跑提交工程（专找 bug）

这一段才是"提交链路"的真验证。按顺序做，**每步都有明确的通过判据**。

### B-0 环境（只做一次）

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
bash scripts/00_setup_env.sh --mode system        # 沿用镜像自带 torch，只补轻量包
python -c "import fastapi, uvicorn, pydantic, requests, yaml, pandas, numpy, scipy, nibabel, SimpleITK, skimage, openpyxl; print('依赖 OK')"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

### B-1 导出提交权重到规范位置

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
mkdir -p /2026aicompetition/workspace/_export_backup
cp -a /2026aicompetition/workspace/checkpoint/. /2026aicompetition/workspace/_export_backup/ 2>/dev/null || true

bash scripts/09_export_submission.sh 0          # 只给折号会补成 g4_fold0
bash scripts/09_export_submission.sh --verify
ls -l /2026aicompetition/workspace/checkpoint/*/
```

**通过判据**：六个目标目录都在，`--verify` 全 `ok`：

```text
goal1_authenticity/model.pt      goal2_stitched/model.pt      goal2_duplicate/encoder.pt
goal3_tumor/model.pt             goal4_diagnosis/model.pt     goal5_segmentation/{core,flair}.pt
```

> `09_export_submission.sh` 是 **`cp -f 源 目标`** 的只读复制（不改源文件），所以安全。
> ⚠️ 但它读的是 live 的 `checkpoints/g4_fold0/best.pth` → 建议在**刚看到一行 epoch 日志之后**立刻导出（那一刻写盘刚结束）。
> ⚠️ `--verify` **只查文件齐不齐，查不出"这是不是 2 轮的演练权重"** → 用 §1 的 `epoch` 自己判断。

### B-2 提交工程契约测试

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 -m pytest tests/ -q                    # 期望全部 passed
```

### B-3 用真实权重跑一次本地推理

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
mkdir -p logs

nohup env COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline \
     COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint \
     python3 scripts/local_eval.py \
       --dataset /2026aicompetition/datasets/verification \
       --output  /2026aicompetition/workspace/answer/local-001 \
       --evaluation-id local-001 > logs/local_eval.log 2>&1 &
echo "PID=$!"
```

> ⚠️ **`local_eval.py` 没有 `--limit`**，会跑完整个 `--dataset`。
> `verification/` 约 777 例 × 全图滑窗 ≈ 20 s/例 → **4 小时以上**，还会和训练抢卡。
> **所以：跑 2~3 分钟就去看格式，格式对了再决定要不要让它跑完（或直接 kill）。**

```bash
sleep 150
tail -20 logs/local_eval.log
ls /2026aicompetition/workspace/answer/local-001 | head

# 抽查一份答案的格式
python3 - <<'PY'
import json, glob
ps = sorted(glob.glob('/2026aicompetition/workspace/answer/local-001/*/prediction.json'))
print("已产出答案:", len(ps))
if ps:
    d = json.load(open(ps[0], encoding='utf-8'))
    print("样例:", ps[0])
    print(json.dumps(d, ensure_ascii=False, indent=1)[:1200])
PY
```

**必须看到的两个细节**

| 看什么 | 通过 | 不通过说明 |
|---|---|---|
| 启动日志 | `[registry] ✓ 真实插件工厂已加载: tasks.real_pipeline` | 若出现 `[registry] ⚠️ 未设置 COMPETITION_PIPELINE_FACTORY → 当前跑的是 Dummy 基线` → 环境变量没生效，**答案全是占位内容** |
| 目录结构 | `answer/local-001/<检查号>/{prediction.json, 掩膜文件...}` | 缺 `prediction.json` 或掩膜 → 记 bug |

**确认格式没问题就可以停掉**（一条命令里跑完整个验证集会很久）：

```bash
kill %1 2>/dev/null || pkill -f "scripts/local_eval.py"
```

### B-4 答案合规校验

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 scripts/validate_output.py \
  --dir /2026aicompetition/workspace/answer/local-001 \
  --expect "$(ls /2026aicompetition/workspace/answer/local-001 | grep -v jsonl | paste -sd, -)"
```

### B-5 复现平台协议（最接近真实测评的一步）

```bash
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
python3 scripts/mock_competition.py \
  --dataset   /2026aicompetition/datasets/verification \
  --workspace /2026aicompetition/workspace --timeout 3600
```

**通过判据**：期望 **5 行 PASS**（协议闭环）。这一步会真正拉起服务、按平台方式下发数据、收回调。

### B-6 提交前总检查

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
python3 scripts/25_verify_tasks_integration.py     # 训练↔推理对接
python3 scripts/26_audit_plugin_completeness.py    # 结构 + API + 同源
bash scripts/23_pre_submit_check.sh                # 期望 15 通过 / 0 失败
```

> 本机可能固定失败一项「权重未导出」—— 做完 B-1 后应转绿。

### ★ 两条服务路径都要测（这是最容易漏的 bug）

平台测评容器的**启动命令**有两种可能，跑的是**两套不同代码**：

| 启动命令 | 跑的是 | 是否需要环境变量 |
|---|---|---|
| `bash /2026aicompetition/workspace/dcs/glioma_track4/scripts/06_platform_serve.sh` | `glioma_track4` 的服务（`src.serving.app`） | ❌ 不需要，**默认就是真实模型** |
| `bash start.sh`（提交工程 Docker 镜像的默认入口） | `Glioma_recognition-main/app/server.py` | ✅ **必须** `COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline` |

**两条都要跑一遍**（B-3 用的是提交工程那条）。第二条漏了环境变量就是"服务一切正常、答案没有意义"的静默失败 —— 启动日志会打印 `[registry]` 那一行供核对，**一定要看**。

```bash
# 测 track4 的服务（平台推荐的那条）
cd /2026aicompetition/workspace/dcs/glioma_track4
nohup bash scripts/06_platform_serve.sh > logs/serve_track4.log 2>&1 &
sleep 20
curl -s http://127.0.0.1:8000/health        # 期望 200 + 健康信息
tail -30 logs/serve_track4.log              # 看「缺失依赖: 无 ✓」与权重发现日志
kill %1 2>/dev/null
```

---

## 4. 共卡资源账（训练还在跑）

| 项 | 显存 | 内存 |
|---|---|---|
| 训练进程（route A） | 12–18 GB | 5–7 GB |
| 单个评估 / 推理进程 | 3–6 GB | 3–5 GB |
| 提交工程 `local_eval.py` | 2–3 GB | 2–4 GB |

合计 **< 30 GB 显存**，32 GB 内存容器也够。但：

- **别并发开多个评估进程**（会互相挤，还可能把训练拖慢到超时）；
- 两个进程抢卡时训练变慢是**正常现象**，不是 bug；
- 想更稳：A 段和 B 段**串行**做，别同时跑。

---

## 5. C 段 · 提交测试结果到平台

> 平台**不是**"上传一个结果文件"。流程是：**你的代码 + 权重放在平台存储里 → 平台拉起你的测评容器 → 通过 8000 端口下发数据 → 你的服务把答案写进 `answer/<evaluation_id>/`**。
> 所以"提交"= 把**代码、权重、启动命令**三件事准备好。

### C-1 三件东西就位

| # | 东西 | 位置 | 命令 / 动作 |
|---|---|---|---|
| 1 | **代码** | Codeup 仓库 → 容器内 `/2026aicompetition/workspace/dcs/` | `git push`（本地）→ 容器内 `git pull` |
| 2 | **权重** | `/2026aicompetition/workspace/checkpoint/<goal>/` | 见 B-1（或路线 B 的 `glioma_goals/scripts/export_to_submission.sh`） |
| 3 | **启动命令** | 训推平台的测评容器配置里填 | 见 C-2 |

⚠️ **两条路线的导出会互相覆盖**（都写同一个 `checkpoint/<goal>/`）。按当前方案（路线 A 折内 8:2）**只保留路线 A 的导出结果**，别再用路线 B 导一遍盖回去。

### C-2 创建测评容器

**路径**：训推平台 → **AI开发 → 容器实例 → 添加实例** → 实例类型选 **测评容器**

> ⚠️ **单租户最多 1 个测评容器**。创建前先清理训练容器腾配额（**先确认代码/权重都在 `workspace/` 下**）。

启动命令填：

```bash
bash /2026aicompetition/workspace/dcs/glioma_track4/scripts/06_platform_serve.sh
```

硬要求（平台会持续探测，不满足则容器起不来）：

| 要求 | 状态 |
|---|---|
| 监听 **8000** 端口 | ✅ `06_platform_serve.sh` / `start.sh` |
| 提供 **`/health`** | ✅ |
| Dockerfile `EXPOSE 8000` | ✅ |

**容器内自检**：

```bash
curl -s http://127.0.0.1:8000/health          # 期望 200
```

**平台侧确认**：容器信息页的 **4 个 Conditions 全为 `true`**。

### C-3 发起测评

| 步骤 | 在哪 | 限制 |
|---|---|---|
| **验证测评**（先跑这个） | 赛事管理平台 → 个人工作台 → 初赛阶段 → 验证测评 → 发起 | **每天 5 次** |
| **初赛测评**（正式） | 同上 → 初赛测评 → 发起 | **每赛道只 1 次**；发起后**冻结部署权限**，不能再改容器配置 |

> ⚠️ **务必先用验证测评确认无误**。初赛测评列表看不到成绩，成绩统一发布后在「**我的成绩**」查看。

### C-4 测评期间怎么看

| 看什么 | 位置 |
|---|---|
| 服务日志 / 回调 | 容器实例 → 点实例 ID → **日志** |
| 答案是否产出 | 容器内 `ls /2026aicompetition/workspace/answer/<evaluation_id>/` |
| 规范 JSONL | `/2026aicompetition/workspace/logs/` |

---

## 6. 找 bug 专用清单（按"静默失败"风险排序）

> "静默失败"= 不报错、流程走完、但结果没有意义。以下每条都必须**主动确认**。

| # | 风险 | 怎么确认 | 症状 |
|---|---|---|---|
| ① | **跑了 Dummy 基线** | 启动日志有 `[registry] ✓ 真实插件工厂已加载` | 服务正常、答案格式合规、**分数极低** |
| ② | **用了演练/冒烟权重** | `torch.load(...)['epoch']` 只有 1~2 → 是演练权重 | `--verify` **查不出**，分数偏低 |
| ③ | **权重是 LFS 指针** | 文件 ~130 字节、前 8 字节 `b'version '` | `invalid load key, 'v'` |
| ④ | **阈值被带偏** | `FOLDS="0" bash scripts/16_finalize.sh --print-split` 应为 `split=oof` | 若出现 `split=external` → `data/manifest_val.json` 还在，**移走** |
| ⑤ | **掩膜写回了错误序列空间** | `data/modality_model.json` 是否随代码入库（`.gitignore` 白名单里） | 评测端直接判错 |
| ⑥ | **三个工程不在同一层** | `ls -d /2026aicompetition/workspace/dcs/*/` | 脚本里写死的相对路径失效 |
| ⑦ | **答案目录缺例** | `validate_output.py`；对比下发例数 | 部分检查号没有答案 |
| ⑧ | **单例脏文件拖垮整批** | `06_platform_serve.sh` 已内置 `GLIOMA_LOADER_TOLERANT=1` | 若见整批失败 → 该开关没生效 |
| ⑨ | **上传时 CRLF 污染脚本** | `bash -n <脚本>` 先验语法 | `bash\r: command not found` |
| ⑩ | **容器删除导致数据丢失** | 代码/权重/`folds.json` 是否都在 `workspace/` 下 | 重启后文件不见了 |

---

## 7. 建议执行顺序（一步一验，别跳）

```text
①  打快照（§1）                                     ← 必须先做
②  A-字段 20 例（§2）                               ← 几十秒，验字段链路
③  A-分割 20 例（§2）                               ← 几分钟，验分割链路
④  B-0 依赖自检 → B-1 导出 + --verify（§3）          ← 秒级，最先出结论
⑤  B-2 pytest 契约测试（§3）                         ← 分钟级
⑥  B-3 local_eval 跑 2 分钟看格式 → 通过则让它跑完或 kill（§3）
⑦  B-4 validate_output → B-5 mock_competition（§3）  ← 期望 5 行 PASS
⑧  B-6 提交前三条检查（§3）
⑨  测 06_platform_serve.sh 的 /health（§3 末）
⑩  训练跑完 → SKIP_MOCK=1 FOLDS="0" bash scripts/16_finalize.sh   ← 收尾（写回阈值，**此刻才能跑**）
⑪  15_eval_special_dup.py --split oof（目标①②，训练后）
⑫  用正式权重重跑 B-1~B-6，再进 C 段发起验证测评
```

---

## 8. 验收清单

**A 段（本地指标）**

- [ ] `35_eval_fields.py` 跑通，14 个字段都有数字（`nan` 只出现在"无负样本"的 `TumorProbability.AUC`）
- [ ] `logs/eval_fold0.log` 有 Dice/NSD/HD95，且 `n_no_mask` 合理
- [ ] 确认用的是**快照**（`$SNAP/best.pth`），不是 live 文件

**B 段（提交工程链路）**

- [ ] `09_export_submission.sh --verify` 六个 goal 全 `ok`
- [ ] `pytest tests/ -q` 全 passed
- [ ] `local_eval.py` 启动日志出现 `[registry] ✓ 真实插件工厂已加载`
- [ ] `answer/local-001/<检查号>/prediction.json` 结构正确（含 `AccessionNumber` + `Prediction` 全字段）
- [ ] `validate_output.py` 通过
- [ ] `mock_competition.py` **5 行 PASS**
- [ ] `23_pre_submit_check.sh` 全绿
- [ ] `06_platform_serve.sh` 起来后 `curl :8000/health` 返回 200

**C 段（平台提交）**

- [ ] 代码已 push 到 Codeup，容器内 `git pull` 到最新
- [ ] `checkpoint/<goal>/` 六个目标权重齐全，且**不是演练权重**（查 `epoch`）
- [ ] 测评容器启动命令已填 `06_platform_serve.sh`（或 `start.sh` **+ FACTORY 环境变量**）
- [ ] 容器信息页 **4 个 Conditions 全 true**
- [ ] 已跑通**验证测评**（每天 5 次）
- [ ] 确认不再改容器配置 → 再发起**初赛测评**（每赛道 1 次，发起后冻结）

---

## 9. 命令速查

| 目的 | 命令 |
|---|---|
| 打快照 | 见 §1（带重试的 `for` 循环） |
| 看训练到第几轮 | `grep -c '^\[g4_fold0\] epoch ' logs/train_fold0.log` |
| 字段指标 | `python3 scripts/35_eval_fields.py --split fold --fold 0 --ckpt "$SNAP" [--limit 20]` |
| 分割指标 | `CKPT="$SNAP" bash scripts/04_eval.sh --split fold 0 [20]` |
| 目标①②+重复 | `python3 scripts/15_eval_special_dup.py --split oof`（训练后） |
| 导出提交权重 | `bash scripts/09_export_submission.sh 0` → `--verify` |
| 契约测试 | `cd Glioma_recognition-main && python3 -m pytest tests/ -q` |
| 本地推理 | 见 §3 B-3 的 `local_eval.py` |
| 答案校验 | `python3 scripts/validate_output.py --dir <answer> --expect "$(ls <answer> \| grep -v jsonl \| paste -sd, -)"` |
| 协议复现 | `python3 scripts/mock_competition.py --dataset <val> --workspace /2026aicompetition/workspace --timeout 3600` |
| 提交前检查 | `bash scripts/23_pre_submit_check.sh` + `25_verify_tasks_integration.py` + `26_audit_plugin_completeness.py` |
| 启动服务自检 | `curl -s http://127.0.0.1:8000/health` |
| 收尾（**训练结束后**） | `SKIP_MOCK=1 FOLDS="0" bash scripts/16_finalize.sh` |

---

## 10. 一句话小结

**A 段**用 track4 的三个脚本 + **模型快照**快速看指标（有 `--limit`，分钟级）；
**B 段**用 `09_export_submission` → `pytest` → `local_eval` → `validate_output` → `mock_competition` 验证**提交链路**（慢，放后台，只看格式与协议合规）；
**C 段**把**代码 + 权重 + 启动命令**三件事就位，创建测评容器，先打**验证测评**（每天 5 次），确认无误再发起**初赛测评**（每赛道 1 次，发起后冻结）。
全程唯一不能提前做的是 **`16_finalize.sh`** —— 它会写回 `best.pth`，必须等训练结束。
