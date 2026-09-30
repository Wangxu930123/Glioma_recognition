# 测评容器操作手册（全命令版 · 可复制粘贴）

> **定位**：从"拿到一个测试容器"到"成绩回到平台"的**全部命令**，按操作顺序排列。
> 每一步自带"期望输出"和"不对时怎么办"的排查命令。遇到任何异常，**先查本节对照表**。
>
> 关联文档：
> - [`INFERENCE_EVAL_AND_SUBMIT_GUIDE.md`](INFERENCE_EVAL_AND_SUBMIT_GUIDE.md) —— 三步提交流程（本手册是其容器侧展开）
> - [`PLATFORM_GUIDE.md`](PLATFORM_GUIDE.md) —— 平台界面零经验版

---

## 0. 容器操作总顺序（先看这张图）

```text
① 拉代码并验脚本          ← 1 分钟     §1
② 检查/修复权重            ← 2 分钟     §2   （LFS 指针是最常见的"跑不起来"）
③ 配置回调地址             ← 1 分钟     §3   （漏了 = 白跑，不计分）
④ 启动服务并确认三行日志    ← 3 分钟     §4
⑤ 发起测评（平台/手动 /call）            §5
⑥ 监控进度                ← 全程       §6
⑦ 成绩回来后：30 秒判定     ← 30 秒      §7
⑧ 异常对照表（遇到问题先翻这里）          §8
```

> 每一段命令**独立可粘贴**，变量都在段内定义。

---

## 1. 拉代码并验脚本

```bash
cd /2026aicompetition/workspace/dcs/glioma_track4 && git pull
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main && git pull

# 工程必须同层（脚本里有写死的相对路径）
ls -d /2026aicompetition/workspace/dcs/*/

# 验脚本没被 CRLF 污染（报 bash\r: command not found 就是它）
bash -n /2026aicompetition/workspace/dcs/glioma_track4/scripts/06_platform_serve.sh && echo "语法 OK"
```

| 期望 | 不对时 |
|---|---|
| `git pull` 显示 up to date 或新提交；`bash -n` 无输出+`语法 OK` | 报 `bash\r` → 本地执行 `git add --renormalize .` 后重新提交，或 `dos2unix` 处理 |

---

## 2. 检查/修复权重（最常见的"跑不起来"都在这）

```bash
# ① 看大小：~130 字节 = LFS 指针（假）；MB/GB 级 = 真权重
ls -l /2026aicompetition/workspace/train_model/g4_fold*/best.pth \
      /2026aicompetition/workspace/dcs/glioma_track4/checkpoints/g4_fold*/best.pth \
      /2026aicompetition/workspace/checkpoint/*/*.pt 2>/dev/null

# ② 逐份真伪验证（读得动 + epoch 不是 1~2 的演练权重）
python3 - <<'PY'
import glob, torch
paths = (glob.glob('/2026aicompetition/workspace/train_model/*/*.pth')
         + glob.glob('/2026aicompetition/workspace/dcs/glioma_track4/checkpoints/g4_fold*/best.pth')
         + glob.glob('/2026aicompetition/workspace/checkpoint/*/*.pt'))
for p in sorted(set(paths)):
    try:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        print(f"OK   epoch={ck.get('epoch')}  {p}")
    except Exception as e:
        print(f"✗ {p}  {repr(e)[:60]}")
PY
```

| 输出 | 处置 |
|---|---|
| 全部 `OK` 且 epoch 合理 | 跳到 §3 |
| 有 `✗ invalid load key, 'v'` | **LFS 指针**：`cd /2026aicompetition/workspace/dcs/glioma_track4 && git lfs pull`，然后重跑上面的验证 |
| 有 `✗` 但不是 `'v'` | 文件截断（传输中断）：回本地重新上传/重新导出 |
| `epoch=1` 或 `2` | 演练权重 —— 能跑但分数低，确认是否要现在测 |

---

## 3. 配置回调地址（漏了 = 推理跑完也不计分）

```bash
# 地址在【容器实例页面】上方，逐字复制（含端口、路径、末尾斜杠）
echo 'http://<容器实例页面上的完整地址>/api/competition/inference/callback/' \
  > /2026aicompetition/workspace/callback_url.txt

# 确认真的写进去了（常见错误：echo 后面少了 >，只打印没写文件）
cat /2026aicompetition/workspace/callback_url.txt

# 5 秒探路：这个地址本身通不通
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  "$(cat /2026aicompetition/workspace/callback_url.txt)" \
  -H 'Content-Type: application/json' -d '{}'
```

| curl 返回码 | 含义 | 处置 |
|---|---|---|
| **404** | **路径不对** | 逐字核对页面上的地址：端口、`/api/competition/inference/callback/`、末尾斜杠 |
| 400 / 405 / 422 | **路径存在**（空 body 被拒是正常） | ✓ 没问题 |
| 超时/拒绝 | 容器与平台网络不通 | 换地址写法或找平台 |

> **提交工程那条服务**（`start.sh`）**只认环境变量、没有文件兜底**，启动前要多加一行：
> `export COMPETITION_CALLBACK_URL="$(cat /2026aicompetition/workspace/callback_url.txt)"`

---

## 4. 启动服务并确认三行日志

```bash
# 平台测评容器：把这一行填进"启动命令"（唯一要填的东西）
bash /2026aicompetition/workspace/dcs/glioma_track4/scripts/06_platform_serve.sh

# 手动测试时后台起：
cd /2026aicompetition/workspace/dcs/glioma_track4
nohup bash scripts/06_platform_serve.sh > logs/serve_manual.log 2>&1 &
sleep 30
tail -40 logs/serve_manual.log
```

**必须看到的三行**（缺一不可）：

```bash
grep -E "启动预检通过|回调地址|推理权重" logs/serve_manual.log | head -5
curl -s http://127.0.0.1:8000/health
```

| 必须出现 | 缺了说明 |
|---|---|
| `[serving] 启动预检通过：权重已加载，服务就绪` | **权重加载失败**（`invalid load key` 等）→ 回 §2。⚠️ 服务仍会起来、/health 仍 200 —— **/health 是假信号**，别被它骗了 |
| `[06] ✓ 回调地址: ...` 或 `[serving] ✓ 回调地址: ...` | 回调没配上 → 回 §3 |
| `[06] 推理权重: ...` | 权重没被发现 → 路径不对，回 §2 的 ① |

---

## 5. 发起测评

**平台正式路径**：赛事管理平台 → 个人工作台 → 初赛阶段 → **验证测评**（每天 5 次）→ 发起。平台自动 `/call`、自动收结果。

**手动测试**（不烧测评次数）：

```bash
EID="manual-$(date +%H%M%S)"    # 每次必须换新 ID（已存在会 FileExistsError）
curl -s -X POST http://127.0.0.1:8000/call \
  -H 'Content-Type: application/json' \
  -d "{\"request_id\":\"$EID\",\"team_id\":\"t\",\"track_code\":\"track4\",
       \"input\":{\"evaluation_id\":\"$EID\",
                  \"dataset_path\":\"/2026aicompetition/datasets/verification/original\"}}"
echo        # 期望 {"code":200,"msg":"accepted","request_id":"..."}
```

| 返回 | 处置 |
|---|---|
| `{"code":200,...}` | 受理成功 → §6 盯进度 |
| `{"code":400,"msg":"缺少 ..."}` | body 字段名不对，对照上面模板 |
| `{"code":400,"msg":"dataset_path 不存在"}` | 换数据根：`ls /2026aicompetition/datasets/` 看真实挂载名 |

---

## 6. 监控进度

```bash
# ① 状态接口（最快）—— running / done / failed
curl -s http://127.0.0.1:8000/status/<request_id>

# ② 实时日志（每例一行「完成」）
tail -f /2026aicompetition/workspace/logs/serving.out

# ③ 跑到第几例了（对照总数 777）
grep -c "完成" /2026aicompetition/workspace/logs/serving.out

# ④ 速率（剩余时间估算：剩余例数 × 单例耗时）
tail -5 /2026aicompetition/workspace/logs/serving.out | grep -o "完成[0-9]*ms"

# ⑤ 答案产出没有（⚠️ 运行中在隐藏目录，要 ls -a）
ls -a /2026aicompetition/workspace/answer/ | head
find /2026aicompetition/workspace/answer -name prediction.json 2>/dev/null | wc -l
```

> **答案目录看起来是空的 ≠ 没产出**：运行中答案写在 `.xxx.tmp-<uuid>` 隐藏目录里，
> 全部跑完才改名为正式的 `answer/<evaluation_id>/`。用上面的 `find` 看真实数量。

---

## 7. 成绩回来后 · 30 秒判定（在答案目录上跑）

> 用途：区分「**没送达**」「**答案本身是 0**」「**跑了 Dummy**」三种一模一样表现为
> "跑完了但 0 分/无成绩"的故障 —— 修法完全不同，判错方向白费时间。

```bash
A=/2026aicompetition/workspace/answer/<那次0分的evaluation_id>
python3 - "$A" <<'PY'
import glob, json, sys
from pathlib import Path
import nibabel as nib
import numpy as np

A = Path(sys.argv[1])
ps = sorted(glob.glob(str(A / "*" / "prediction.json")))
print(f"答案数: {len(ps)}")

p_zero = p_total = m_empty = m_total = 0
samples = []
for p in ps[:120]:
    d = json.load(open(p, encoding="utf-8"))
    tp = (d.get("Prediction") or {}).get("TumorProbability")
    if tp is not None:
        p_total += 1
        p_zero += (tp == 0.0)
        if len(samples) < 5: samples.append(tp)
    acc = Path(p).parent
    for role, uri in (d.get("SegmentationMaskURI") or {}).items():
        try:
            v = np.asanyarray(nib.load(str(acc / uri.lstrip("./"))).dataobj)
        except Exception as e:
            print("  掩膜读不了:", uri, repr(e)[:50]); continue
        m_total += 1
        m_empty += (int(v.sum()) == 0)

print(f"TumorProbability=0.0 的: {p_zero}/{p_total}   样值: {samples}")
print(f"掩膜全空的: {m_empty}/{m_total}")

if p_total and p_zero >= p_total * 0.95:
    print(">>> 判定C: 概率全 0 → 跑的是 Dummy 基线（工厂变量没设）")
elif m_total and m_empty >= m_total * 0.5:
    print(">>> 判定B: 掩膜过半为空 → 代码是修复前的旧版，git pull 重跑")
elif p_total and m_total and p_zero < p_total * 0.5 and m_empty < m_total * 0.2:
    print(">>> 判定A: 答案健康 → 0 分出在送达：修回调即可，推理链路没问题")
PY
```

**判定后的动作**：

| 判定 | 动作 |
|---|---|
| **A（没送达）** | 回 §3 重配回调；确认闭环：`grep -o '"callback":"[^"]*"' /2026aicompetition/workspace/logs/serving.jsonl \| tail -3` —— **有 `"callback":"ok"` 才闭环**；然后重发测评 |
| **B（旧代码）** | `git log -1 --format='%h %s %ci'`（两个仓都查）→ 不是最新就 `git pull` + 重启服务 + 重测 |
| **C（Dummy）** | 服务日志查 `grep "未设置 COMPETITION_PIPELINE_FACTORY" logs/*.log` → 用 `./start.sh` 启动或补 `export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline` |

---

## 8. 异常对照表（遇到问题先翻这里）

### 8.1 重跑报 `evaluation output already exists`

```text
FileExistsError: evaluation output already exists: .../answer/<evaluation_id>
```

**原因**：`evaluation_id` 用过了（答案目录已存在，Writer 拒绝覆盖）。
**修法**：换个新 ID 重发（§5 的 `EID="manual-$(date +%H%M%S)"` 就是为此）；或确认旧答案不要了再删：

```bash
rm -rf /2026aicompetition/workspace/answer/<旧evaluation_id>
```

### 8.2 `invalid load key, 'v'`

权重是 **LFS 指针**（~130 字节的文本）。回 §2。

### 8.3 推理报 `callback failed after 3 attempts: HTTP Error 404`

回调 URL 路径不对。回 §3 的探路命令，404 = 地址错（逐字核对页面上的完整地址）。

### 8.4 推理 failed：`KeyError: "权重中不存在分类字段 ..."`

```bash
curl -s http://127.0.0.1:8000/status/<request_id>   # 看 error 全文
tail -30 /2026aicompetition/workspace/logs/serving.out
```

**原因**：权重与配置不匹配（单任务研发权重 vs 多任务服务）→ 用 `09_export_submission.sh` 导出的**多任务权重**替换，见 §2。

### 8.5 第一例就 OOM / 容器被平台杀（"内存占用高已停止容器实例"）

```bash
# 先看是不是病态网格（代码里有护栏，会明确报错而不是让容器死）
grep -E "公共网格|GLIOMA_MAX_GRID_VOXELS" /2026aicompetition/workspace/logs/serving.out | tail -5

# 确认 GPU 可见性（GPU 不可见时全部激活落 RAM，本就是 GB 级）
python3 -c "import torch; print('cuda', torch.cuda.is_available())"

# 逐阶段定位峰值（追踪器，容器被杀时最后一行 → 就是元凶阶段）
cd /2026aicompetition/workspace/dcs/Glioma_recognition-main
COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint \
python3 scripts/stage_mem_trace.py 3 2>&1 | tee /tmp/stage_trace.log
```

| 日志停在 | 元凶 |
|---|---|
| `→ load <acc>` | 该例原始 NIfTI 过大 |
| `→ <acc> goal5` | 公共网格爆炸（看护栏报错）/ 滑窗 / 后处理 |
| `→ build_pipeline` | 权重目录里 `*.pt` 太多，常驻模型装不下 |

### 8.6 `/status` 返回 `unknown`

`/call` 没被受理（request_id 不对或 /call 失败）。回 §5 看那次 `/call` 的返回体。

### 8.7 `inference.jsonl` 里 request_id 对不上

**无害**：追加式日志，`1`、`2`、`09272034` 是**不同次运行**的记录。只看当前这次：

```bash
grep '"request_id":"<这次的>"' /2026aicompetition/workspace/logs/inference.jsonl
```

### 8.8 平台显示测评完成但没成绩

```bash
# ① 答案真的产出了吗
ls /2026aicompetition/workspace/answer/<evaluation_id>/ | wc -l

# ② 回调闭环了吗（关键：有 "ok" 才算送达）
grep -o '"callback":"[^"]*"' /2026aicompetition/workspace/logs/serving.jsonl | tail -3

# ③ 本次 evaluation 的结局（jsonl 里最后几条）
grep '"evaluation_id":"<这次的>"' /2026aicompetition/workspace/logs/inference.jsonl | tail -3
```

| ② 的输出 | 结论 |
|---|---|
| `"callback":"ok"` | 已送达 → 没成绩是平台侧延迟/统计口径，等发布 |
| `FAILED_no_callback_url` | §3 重配 |
| `error:HTTP ...` | §3 探路修 404 |

### 8.9 单例失败会拖垮整批吗

不会 —— `06_platform_serve.sh` 已 `GLIOMA_LOADER_TOLERANT=1`：脏数据只跳过该例。
**验证它生效**：日志搜 `[runner][容错] 跳过检查`，被跳过的检查号会列出、其余照常产出。

### 8.10 想重跑同一次测评

平台侧：直接再发起一次（验证测评每天 5 次）。手动侧：**必须换新 evaluation_id**（见 8.1）。

---

## 9. 命令速查（全表）

| 目的 | 命令 |
|---|---|
| 拉代码 | `cd <工程> && git pull`（两个仓都拉） |
| 验脚本 CRLF | `bash -n scripts/06_platform_serve.sh` |
| 权重大小 | `ls -l checkpoints/g4_fold*/best.pth` |
| 权重真伪 | §2 的 heredoc（逐份 torch.load + epoch） |
| 修 LFS | `git lfs pull` |
| 写回调文件 | `echo '<URL>' > /2026aicompetition/workspace/callback_url.txt` |
| 探回调 URL | `curl -s -o /dev/null -w '%{http_code}' -X POST "$(cat .../callback_url.txt)" -H 'Content-Type: application/json' -d '{}'` |
| 起服务 | `nohup bash scripts/06_platform_serve.sh > logs/serve.log 2>&1 &` |
| 探活 | `curl -s http://127.0.0.1:8000/health` |
| 看启动预检 | `grep "启动预检通过" logs/serve.log` |
| 发起推理 | §5 的 `/call` 模板（evaluation_id 每次换新） |
| 查状态 | `curl -s http://127.0.0.1:8000/status/<request_id>` |
| 跟日志 | `tail -f /2026aicompetition/workspace/logs/serving.out` |
| 进度计数 | `grep -c "完成" .../logs/serving.out` |
| 答案计数（含隐藏 staging） | `find .../answer -name prediction.json \| wc -l` |
| 30 秒判定 | §7 的 heredoc |
| 回调闭环确认 | `grep -o '"callback":"[^"]*"' .../logs/serving.jsonl \| tail -3` |
| 看本次 jsonl | `grep '"request_id":"<id>"' .../logs/inference.jsonl` |
| 删旧答案（重跑前） | `rm -rf .../answer/<旧evaluation_id>` |
| OOM 逐阶段定位 | `python3 scripts/stage_mem_trace.py 3` |
| 删旧答案目录（谨慎） | 8.1 —— 确认不要了再删 |

---

## 10. 一句话

**顺序**：§1 拉代码 → §2 权重 → §3 回调 → §4 三行日志 → §5 发起 → §6 盯进度 →
§7 判定。异常先翻 §8（十种已知情况都有对应命令），全部命令在 §9。
