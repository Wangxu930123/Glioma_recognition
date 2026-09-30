# 推理测评 + 提交测试结果到平台 手册（精简版）

> **定位**：不做任何本地测试 —— **直接在测评容器跑**。全文只保留三件必须的事：
> ① 权重就位 ② 创建测评容器（一行启动命令）③ 发起测评收成绩。
>
> 关联文档：
> - [`PLATFORM_GUIDE.md`](PLATFORM_GUIDE.md) —— 云桌面 / 训推平台零经验版（界面路径看它）
> - [`ZERO_SCORE_TROUBLESHOOT.md`](ZERO_SCORE_TROUBLESHOOT.md) —— **测评容器操作手册（全命令版）**：容器内操作顺序、30 秒判定、十种异常对照与排查命令
> - [`MIGRATE_AND_RESUME_GUIDE.md`](MIGRATE_AND_RESUME_GUIDE.md) —— 代码搬迁与权重真伪判定
> - [`OOM_AND_RESUME_GUIDE.md`](OOM_AND_RESUME_GUIDE.md) —— 内存/断点续训
>
> 所有命令都能整段粘贴。

---

## 0. 三十秒速览（三步走完）

```text
第 1 步（一次性，几分钟）  代码 + 权重就位
    git push → 容器内 git pull；权重导出到 /2026aicompetition/workspace/checkpoint/<goal>/   ← §2

第 2 步（创建一次）        训推平台 → AI开发 → 容器实例 → 添加实例 → 实例类型选「测评容器」
    启动命令填一行：
      bash /2026aicompetition/workspace/dcs/glioma_track4/scripts/06_platform_serve.sh          ← §3
    等容器信息页 4 个 Conditions 全 true

第 3 步（一键测评）        赛事管理平台 → 个人工作台 → 初赛阶段 → 「验证测评」发起             ← §4
    → 平台自动下发数据、你的服务自动推理+写答案+回调
    → 结果返回在平台「我的成绩」
    先用验证测评（每天 5 次）确认无误 → 再发起初赛测评（每赛道 1 次，发起后冻结）
```

> **为什么不在本地跑测试**：平台测评本身就是"一键启动 + 一键测评 + 返回结果"。
> 触发测评、收结果都是**平台**做的，你在容器里唯一要"一键"的就是那行启动命令。

**三条铁律**

1. **导出权重前看一眼训练日志** —— 训练每轮都在重写 `best.pth`，而 `torch.save` **不是原子写**；在刚打出一行 `epoch` 日志之后导出（那一刻写盘刚结束），见 §2。
2. **`06_platform_serve.sh` 是唯一要填的启动命令** —— 它自带真实模型、容错开关与回调地址解析，不需要配任何环境变量。
3. **初赛测评每赛道只 1 次、发起后冻结** —— 务必先用**验证测评**（每天 5 次）确认无误。

---

## 1. 第 1 步 · 代码就位

| # | 东西 | 位置 | 动作 |
|---|---|---|---|
| 1 | **代码** | Codeup 仓库 → 容器内 `/2026aicompetition/workspace/dcs/` | 本地 `git push` → 容器内 `git pull` |

```bash
# 容器内：拉最新代码
cd /2026aicompetition/workspace/dcs/glioma_track4 && git pull
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main && git pull
```

> ⚠️ 三个工程必须在同一层：`ls -d /2026aicompetition/workspace/dcs/*/` —— 脚本里有写死的相对路径。
> ⚠️ **上传时 CRLF 会污染脚本**：容器内先验语法 `bash -n scripts/06_platform_serve.sh`，
> 报 `bash\r: command not found` 就是 CRLF 进来了。

---

## 2. 第 1 步（续）· 导出提交权重到规范位置（必须）

平台推理服务按固定路径找权重：`/2026aicompetition/workspace/checkpoint/<goal>/`。

**先确认训练刚写完盘**（`torch.save` 非原子，写盘瞬间读会得到半个文件）：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
tail -3 logs/train_fold0.log        # 看到一行完整的 epoch 日志再导出
```

**导出 + 自检**：

```bash
# 备份现有导出（两条路线的导出会互相覆盖同一个 checkpoint/，留一份保险）
mkdir -p /2026aicompetition/workspace/_export_backup
cp -a /2026aicompetition/workspace/checkpoint/. /2026aicompetition/workspace/_export_backup/ 2>/dev/null || true

bash scripts/09_export_submission.sh 0          # 只给折号会补成 g4_fold0
bash scripts/09_export_submission.sh --verify  # 自检：六个目标全 ok
ls -l /2026aicompetition/workspace/checkpoint/*/
```

**通过判据** —— 六个目标目录都在，`--verify` 全 `ok`：

```text
goal1_authenticity/model.pt      goal2_stitched/model.pt      goal2_duplicate/encoder.pt
goal3_tumor/model.pt             goal4_diagnosis/model.pt     goal5_segmentation/{core,flair}.pt
```

**再确认这不是演练权重**（`--verify` 只查文件齐不齐，查不出"这是几轮的模型"）：

```bash
python3 - <<'PY'
import torch, glob
for p in sorted(glob.glob('/2026aicompetition/workspace/checkpoint/*/*.pt')):
    try:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        print(f"{ck.get('epoch', '?'):>4} epoch  {p}")
    except Exception as e:
        print(f"  ✗ 读不动: {p}  ({e})")
PY
```

| 看到 | 含义 |
|---|---|
| `epoch = 1~2` | 演练/冒烟权重 —— 分数必然偏低，确认是否要现在测 |
| 文件 ~130 字节、报 `invalid load key, 'v'` | **LFS 指针**，不是权重 —— 回本地 `git lfs pull` |
| `epoch` 合理且能正常 torch.load | ✅ |

> ⚠️ 两条路线（A 折内 8:2 / B）的导出**互相覆盖**同一个 `checkpoint/<goal>/`。
> 按当前方案**只保留路线 A 的导出结果**，别再用路线 B 导一遍盖回去。
> ⚠️ **`16_finalize.sh` 现在绝对不要跑** —— 它会写回阈值、改 `best.pth`，必须等训练结束。

---

## 3. 第 2 步 · 创建测评容器（一行启动命令 = 一键启动）

**路径**：训推平台 → **AI开发 → 容器实例 → 添加实例** → 实例类型选 **测评容器**

> ⚠️ **单租户最多 1 个测评容器**。创建前先清理训练容器腾配额（**先确认代码/权重都在 `workspace/` 下** —— 容器删除后非 workspace 的东西会丢）。

**启动命令**填（就这一行）：

```bash
bash /2026aicompetition/workspace/dcs/glioma_track4/scripts/06_platform_serve.sh
```

这个脚本**自动完成**（不需要你配任何环境变量）：

| 它做的事 | 说明 |
|---|---|
| 找权重 | `$WS/train_model/g4_fold*/best.pth` → `checkpoints/g4_fold*/best.pth`（多折逗号拼接=集成） |
| 开容错 | `GLIOMA_LOADER_TOLERANT=1` —— 单个脏文件只跳过该例，不会让**整批** 0 分 |
| 解析回调地址 | 环境变量（`CALLBACK_URL` 等）→ `$WS/callback_url.txt`；缺了会**大声告警**，因为它=该次测评不计分 |
| 依赖自检 | 缺包会打印清单（先跑 `bash scripts/00_setup_env.sh --mode system` 补齐） |
| 前台常驻 | 8000 端口 + `/health` + `/call` —— 平台硬性要求 |

**平台硬性要求**（不满足则容器起不来）：

| 要求 | 状态 |
|---|---|
| 监听 **8000** 端口 | ✅ 脚本内置 |
| 提供 **`/health`** 常活（启动探针 5s/次，探活 30s/次） | ✅ |
| `/call` 5s 内回 200（推理在后台异步跑） | ✅ |

**创建后确认**：

1. 容器信息页的 **4 个 Conditions 全为 `true`**
2. 容器日志里能看到 `[06] ✓ 回调地址: ...` —— **看到这行才闭环**；没有就按脚本打印的两种办法补上（写到 `$WS/callback_url.txt` 后重启容器），否则**推理跑完也不计分**

> ⚠️ **回调地址是"该次测评计不计分"的开关**，地址见容器实例页面上方。

---

## 4. 第 3 步 · 发起测评（一键测评，平台自动跑完并返回成绩）

| 步骤 | 在哪 | 限制 |
|---|---|---|
| **验证测评**（先跑这个） | 赛事管理平台 → 个人工作台 → 初赛阶段 → 验证测评 → 发起 | **每天 5 次** |
| **初赛测评**（正式） | 同上 → 初赛测评 → 发起 | **每赛道只 1 次**；发起后**冻结部署权限**，不能再改容器配置 |

发起后平台自动：下发数据 → 你的服务推理（约 6~8 s/例 × 777 例 ≈ 1.5~2 小时）→ 写答案 → 回调 → 出分。

**期间怎么看**：

| 看什么 | 位置 |
|---|---|
| 服务日志 / 回调 | 容器实例 → 点实例 ID → **日志** |
| 进度（每例一行「开始/完成」） | 日志里 `[runner] #N <检查号> 完成 xxxms ...` |
| 答案是否产出 | 容器内 `ls /2026aicompetition/workspace/answer/<evaluation_id>/`（注意：跑完才改名为正式目录，运行中在 `.xxx.tmp-*` 隐藏目录里，要 `ls -a`） |
| **成绩** | 统一发布后在「**我的成绩**」查看（初赛测评列表里看不到） |

**结果异常时对照**（见 §5 排雷表）。

### 4b. 手动模拟平台测评（两个终端 + curl，不烧测评次数）

> 平台测评用的就是这套 HTTP 协议。先手动走一遍 = 零成本彩排。
> 下方流程已修正五个已知坑（响应体、ID 复用、路径、精度、ID 记录），可直接照抄。

**终端 1 · 启动系统**（回调地址见容器实例页面上方，逐字复制）：

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4
CALLBACK_URL='http://<容器实例页面上的完整地址>/api/competition/inference/callback/' \
  nohup bash scripts/06_platform_serve.sh > logs/serve.log 2>&1 &
sleep 30
# 确认三行（/health 是假信号——它只查环境变量非空；真判定是「启动预检通过」）
grep -E "启动预检通过|✓ 回调地址|推理权重" logs/serve.log | head -5
curl -s http://127.0.0.1:8000/health
```

**终端 2 · 发起**（ID 每次换新；`dataset_path` 换成真实挂载）：

```bash
EID="$(date +%m%d%H%M%S)$RANDOM"          # 天然不重复的 evaluation_id
echo "request_id=$EID-req  evaluation_id=$EID   ← 记下来，/status 用前者、答案目录用后者"

curl -X POST http://127.0.0.1:8000/call \
  -H 'Content-Type: application/json' \
  -d "{
    \"request_id\":\"$EID-req\",
    \"team_id\":\"2054451790241292288\",
    \"track_code\":\"2eaf6e36583b4d06af0f4582220956d0\",
    \"input\":{
      \"evaluation_id\":\"$EID\",
      \"dataset_path\":\"/2026aicompetition/datasets/verification/original\"
    }
  }"
```

**响应体**（修正后会直接告诉你下一步）：

```json
{"code":200,"msg":"accepted","request_id":"...","evaluation_id":"...",
 "status_url":"/status/<request_id>",
 "hint":"推理在后台执行：GET /status/<request_id> 轮询到 done 或 failed 才有结论"}
```

**终端 3（或稍后）· 轮询到有结论** —— ⚠️ **200 只是"已受理"**，推理在后台线程，
失败的报错只在 `/status` 和日志里：

```bash
curl -s http://127.0.0.1:8000/status/$EID-req
#   {"status":"running"}   → 在跑（777 例约 1.5~2 小时），期间可看进度：
#     grep -c "完成" /2026aicompetition/workspace/logs/serving.out
#   {"status":"done","out":"..."}      → 成功，验收 ↓
#   {"status":"failed","error":"..."}  → 失败，error 就是原因（对照 §5/§ZERO_SCORE_TROUBLESHOOT）
```

**done 后验收三件事**：

```bash
ls /2026aicompetition/workspace/answer/$EID/ | head        # 答案目录
find /2026aicompetition/workspace/answer/$EID -name prediction.json | wc -l
grep -o '"callback":"[^"]*"' /2026aicompetition/workspace/logs/serving.jsonl | tail -3
#   "callback":"ok" 才是真的闭环送达；FAILED_no_callback_url / error:HTTP... → §3 重配
```

**这套流程修掉的五个坑**：

| # | 坑 | 修法（代码 / 流程各一半） |
|---|---|---|
| 1 | **`/call` 200 被误读成"测评成功"** —— 失败只出现在后台日志 | 代码：响应体现在带 `status_url` + `hint`；流程：必须轮询 `/status` 到 done/failed |
| 2 | **evaluation_id 复用 → 新旧答案静默混写**同一目录 | 代码：`/call` 受理前拦截（非空目录直接 400，附清理命令）；流程：`EID="$(date ...)…"` 天然不重复 |
| 3 | `dataset_path` 填了占位路径 | 流程：发之前 `ls` 一下真实挂载；400 会即时报错，无害但别浪费一轮 |
| 4 | **`evaluationId` 数值化超 JS 安全整数（2^53-1）被平台截断** → 平台对不上 | 代码：只在安全范围内转数值，超范围保持字符串；流程：evaluation_id 控制在 **15 位内**最稳 |
| 5 | 两个 ID 混淆（`request_id` 查状态、`evaluation_id` 是答案目录名） | 流程：发的时候 `echo` 打印并记下来；代码：响应体两个字段都回显 |

---

## 5. 排雷表（按"静默失败"风险排序 —— 测评结果不对时先查这些）

> "静默失败"= 不报错、流程走完、但结果没有意义。这些**都在第 1、2 步就能提前确认**。

| # | 风险 | 在哪确认 | 症状 |
|---|---|---|---|
| ① | **权重没导出 / 演练权重** | §2 的 `--verify` + `epoch` 自检 | 服务起不来（FileNotFoundError）或分数极低 |
| ② | **权重是 LFS 指针** | §2 的 torch.load 自检 | `invalid load key, 'v'` |
| ③ | **回调地址没配** | 容器日志 `[06] ✓ 回调地址:` | 推理跑完但**不计分** |
| ④ | **代码不是最新** | §1 `git pull` | 修过的 bug 又出现 |
| ⑤ | **脚本被 CRLF 污染** | `bash -n scripts/06_platform_serve.sh` | `bash\r: command not found`，容器起不来 |
| ⑥ | **三个工程不在同一层** | `ls -d /2026aicompetition/workspace/dcs/*/` | 相对路径失效 |
| ⑦ | **`data/modality_model.json` 没随代码入库** | `git ls-files data/modality_model.json` | 模态识别退回内嵌系数（一致率≈27%） |
| ⑧ | **单例脏文件拖垮整批** | `06_platform_serve.sh` 已内置容错开关 | 若整批失败 → 容器没带最新代码 |
| ⑨ | **训练没结束就跑了 `16_finalize.sh`** | 本手册通篇提醒 | 阈值被写偏、`best.pth` 被抢写 |

---

## 6. 验收清单（发起前过一遍）

**第 1 步（代码 + 权重）**

- [ ] 容器内 `git pull` 到最新，三个工程在 `/2026aicompetition/workspace/dcs/` 同层
- [ ] `09_export_submission.sh --verify` 六个 goal 全 `ok`
- [ ] 逐份 `torch.load` 能读、`epoch` 不是 1~2、不是 LFS 指针（§2 自检脚本）
- [ ] `data/modality_model.json` 在仓库里

**第 2 步（测评容器）**

- [ ] 实例类型 = **测评容器**，启动命令 = `bash .../06_platform_serve.sh`（就这一行）
- [ ] 4 个 Conditions 全 `true`
- [ ] 容器日志有 `[06] ✓ 回调地址: ...`
- [ ] 日志有 `[06] 推理权重: ...`（权重发现了）

**第 3 步（测评）**

- [ ] 先发起**验证测评**并等到成绩
- [ ] 成绩正常 → 确认不再改容器配置 → 发起**初赛测评**（1 次，冻结）
- [ ] 训练还没结束的话：**没跑过 `16_finalize.sh`**

---

## 7. 命令速查

| 目的 | 命令 |
|---|---|
| 拉代码 | `cd <工程> && git pull`（三个工程都要） |
| 看训练写完盘没有 | `tail -3 logs/train_fold0.log` |
| 导出 + 自检 | `bash scripts/09_export_submission.sh 0` → `bash scripts/09_export_submission.sh --verify` |
| 权重真伪自检 | §2 的 `torch.load` heredoc 片段 |
| 验脚本没被 CRLF 污染 | `bash -n scripts/06_platform_serve.sh` |
| 补依赖 | `bash scripts/00_setup_env.sh --mode system` |
| 容器内探活 | `curl -s http://127.0.0.1:8000/health` |
| **手动发起测评**（不烧次数） | §4b —— 生成新 `EID` → `POST /call` → 轮询 `/status/$EID-req` 到 done |
| 查一次请求的状态 | `curl -s http://127.0.0.1:8000/status/<request_id>` |
| 回调闭环确认 | `grep -o '"callback":"[^"]*"' /2026aicompetition/workspace/logs/serving.jsonl \| tail -3` |
| 看答案（跑完后） | `ls /2026aicompetition/workspace/answer/<evaluation_id>/` |
| 收尾（**训练结束后才许跑**） | `SKIP_MOCK=1 FOLDS="0" bash scripts/16_finalize.sh` |

---

## 8. 一句话小结

**三步**：① `git pull` + `09_export_submission.sh` 导出权重并自检 →
② 平台创建**测评容器**，启动命令填一行 `06_platform_serve.sh`，确认回调地址行 →
③ 发起**验证测评**（每天 5 次）→ 成绩回「我的成绩」→ 无误再发**初赛测评**（1 次，冻结）。
全程**不在本地跑任何测试**；唯一不能提前做的是 `16_finalize.sh` —— 必须等训练结束。
