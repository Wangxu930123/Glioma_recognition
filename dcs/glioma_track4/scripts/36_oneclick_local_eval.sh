#!/usr/bin/env bash
# ============================================================================
# 36_oneclick_local_eval.sh —— 测试容器内**一键**完成：
#   权重快照 → 起推理服务 → 触发推理（真实权重 × 真实数据）→ 拿回结果 → 验收
#
# 与既有脚本的关系：
#   · 06_platform_serve.sh ：只起服务（不触发推理、无回调闭环）
#   · 08_mock_competition.sh：协议闭环，但用**合成数据**（验协议，不验真权重真数据）
#   · 本脚本             ：**真实权重 × 真实 verification 数据**的完整彩排，
#                          结果通过 mock 回调（scripts/mock_callback.py）拿回。
#
# 用法（在测试容器里，一条命令）：
#   bash scripts/36_oneclick_local_eval.sh                # 前 3 例冒烟（约 2~3 分钟）
#   bash scripts/36_oneclick_local_eval.sh --mini 10      # 前 10 例
#   bash scripts/36_oneclick_local_eval.sh --full         # 全量 777 例（约 90 分钟）
#   bash scripts/36_oneclick_local_eval.sh --keep         # 跑完不杀服务（便于继续手工调试）
#   GLIOMA_CKPT=/abs/a.pth bash scripts/36_oneclick_local_eval.sh   # 显式指定权重
#
# 退出码：0 = 全流程通过（推理 done + 校验 ok + 回调收到 + 答案可读）；1 = 失败
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
WS="${WORKSPACE:-/2026aicompetition/workspace}"
LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"

# ----------------------------- 参数 -----------------------------
MINI=3; FULL=0; KEEP=0; DATASET=""; EVAL_ID="local-oneclick-$(date +%H%M%S)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mini)   MINI="$2"; shift 2 ;;
    --full)   FULL=1; shift ;;
    --keep)   KEEP=1; shift ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --eval-id) EVAL_ID="$2"; shift 2 ;;
    *) echo "[oneclick] ✗ 未知参数: $1（支持 --mini N / --full / --keep / --dataset / --eval-id）"; exit 2 ;;
  esac
done
DATA_ROOT="${DATASET:-/2026aicompetition/datasets/verification/original}"
[ -d "$DATA_ROOT" ] || DATA_ROOT="${DATASET:-/2026aicompetition/datasets/verification}"
[ -d "$DATA_ROOT" ] || { echo "[oneclick] ✗ 数据根不存在: $DATA_ROOT"; exit 2; }

PIDS=()
cleanup() {
  if [[ $KEEP -eq 1 ]]; then
    echo "[oneclick] --keep：保留服务进程（${PIDS[*]:-无}），手工清理请执行："
    echo "  pkill -f 'src.serving.app'; pkill -f mock_callback.py"
    return
  fi
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  echo "[oneclick] 已清理后台进程"
}
trap cleanup EXIT

# ----------------------------- ① 权重快照 -----------------------------
# 训练每轮重写 best.pth 且 torch.save 非原子 —— 必须先快照再推理。
SNAP="$WS/_snap"; mkdir -p "$SNAP"
if [[ -n "${GLIOMA_CKPT:-}" ]]; then
  CKPTS="$GLIOMA_CKPT"
  echo "[oneclick] ① 权重（GLIOMA_CKPT 显式指定）: $CKPTS"
else
  CKPTS=""
  for f in $WS/train_model/g4_fold*/best.pth $ROOT/checkpoints/g4_fold*/best.pth; do
    [ -f "$f" ] || continue
    tag="$(basename "$(dirname "$f")")_$(md5sum "$f" | cut -c1-6)"
    cp -f "$f" "$SNAP/${tag}.pth"
    if python3 -c "import torch,sys;ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False);assert ck.get('model') or ck.get('model_ema');print('[oneclick]   快照OK',sys.argv[1],'epoch=',ck.get('epoch'),'metric=',ck.get('best_metric'))" "$SNAP/${tag}.pth" 2>>"$LOG_DIR/snap_err.log"; then
      CKPTS="${CKPTS:+$CKPTS,}$SNAP/${tag}.pth"
    else
      echo "[oneclick]   ⚠️ 快照损坏（训练正在写入？），跳过 $f"
    fi
  done
  [ -n "$CKPTS" ] || { echo "[oneclick] ✗ 没有可用权重（$WS/train_model/ 或 checkpoints/）；或用 GLIOMA_CKPT= 指定"; exit 2; }
  echo "[oneclick] ① 权重快照: $CKPTS"
fi

# ----------------------------- ② 数据集 -----------------------------
if [[ $FULL -eq 1 ]]; then
  RUN_DS="$DATA_ROOT"
  N_CASES="$(ls -1 "$DATA_ROOT" | wc -l)"
  echo "[oneclick] ② 数据集（全量）: $RUN_DS（$N_CASES 例）"
else
  RUN_DS="/tmp/oneclick_ds_${EVAL_ID}"; rm -rf "$RUN_DS"; mkdir -p "$RUN_DS"
  i=0
  for d in "$DATA_ROOT"/*/; do
    [ -d "$d" ] || continue
    cp -r "$d" "$RUN_DS/"; i=$((i+1))
    [ "$i" -ge "$MINI" ] && break
  done
  N_CASES="$i"
  [ "$N_CASES" -gt 0 ] || { echo "[oneclick] ✗ 数据根下没有病例目录: $DATA_ROOT"; exit 2; }
  echo "[oneclick] ② 数据集（前 $N_CASES 例）: $RUN_DS"
fi

# ----------------------------- ③ mock 回调接收器 -----------------------------
echo "[oneclick] ③ 启动 mock 回调接收器（代替平台收 predPath）…"
pkill -f "mock_callback.py" 2>/dev/null || true
python3 scripts/mock_callback.py --port 9000 > "$LOG_DIR/mock_callback.log" 2>&1 &
PIDS+=($!)
sleep 2
curl -sf http://127.0.0.1:9000/health > /dev/null || { echo "[oneclick] ✗ mock 回调接收器没起来，看 $LOG_DIR/mock_callback.log"; exit 2; }
echo "[oneclick]   ✓ http://127.0.0.1:9000 （收到的回调可用 GET /received 查看）"

# ----------------------------- ④ 推理服务 -----------------------------
echo "[oneclick] ④ 启动推理服务（06_platform_serve.sh + 快照权重 + mock 回调）…"
pkill -f "src.serving.app" 2>/dev/null || true; sleep 1
GLIOMA_CKPT="$CKPTS" \
CALLBACK_URL="http://127.0.0.1:9000/api/competition/inference/callback/" \
  bash scripts/06_platform_serve.sh > "$LOG_DIR/serve_oneclick.log" 2>&1 &
PIDS+=($!)

# 等待「启动预检通过」——不是 /health（那只查环境变量非空，是假信号）
echo "[oneclick]   等待权重加载（最多 300s，判定行=「启动预检通过」）…"
READY=0
for i in $(seq 1 60); do
  if grep -q "启动预检通过" "$LOG_DIR/serve_oneclick.log" 2>/dev/null; then READY=1; break; fi
  if grep -qE "启动预检失败" "$LOG_DIR/serve_oneclick.log" 2>/dev/null; then break; fi
  sleep 5
done
if [[ $READY -ne 1 ]]; then
  echo "[oneclick] ✗ 服务未就绪，最后 30 行日志："; tail -30 "$LOG_DIR/serve_oneclick.log"; exit 2
fi
grep -E "✓ 回调地址|推理权重|缺失依赖|torch .* cuda" "$LOG_DIR/serve_oneclick.log" | head -5
echo "[oneclick]   ✓ 服务就绪（权重已加载）"

# ----------------------------- ⑤ 触发推理 -----------------------------
REQ_ID="req-${EVAL_ID}"
echo "[oneclick] ⑤ POST /call（真实推理开始）…"
HTTP=$(curl -s -o /tmp/oneclick_call.json -w '%{http_code}' -X POST http://127.0.0.1:8000/call \
  -H 'Content-Type: application/json' \
  -d "{\"request_id\":\"$REQ_ID\",\"team_id\":\"local\",\"track_code\":\"track4\",
       \"input\":{\"evaluation_id\":\"$EVAL_ID\",\"dataset_path\":\"$RUN_DS\"}}")
cat /tmp/oneclick_call.json; echo
[ "$HTTP" = "200" ] || { echo "[oneclick] ✗ /call 返回 $HTTP（5s 超时内应答 200）"; exit 2; }

# ----------------------------- ⑥ 轮询直到完成 -----------------------------
EST=$(( N_CASES * 8 + 300 ))                    # 每例 ~8s + 启动余量
TIMEOUT=$(( ${ONECLICK_TIMEOUT:-$EST} ))
echo "[oneclick] ⑥ 轮询推理进度（$N_CASES 例，预计 ≤ $((TIMEOUT/60)) 分钟，Ctrl-C 可中断且自动清理）…"
T0=$(date +%s); STATUS=""
while true; do
  STATUS=$(curl -s "http://127.0.0.1:8000/status/$REQ_ID" | python3 -c "import json,sys
d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo poll_err)
  case "$STATUS" in
    done|failed) break ;;
    unknown) echo "[oneclick] ✗ /status 返回 unknown（/call 没被受理？）"; exit 2 ;;
    poll_err) : ;;
    *)  ELAPSED=$(( $(date +%s) - T0 ))
        printf "\r[oneclick]   %-9s %3ds / %ds" "$STATUS" "$ELAPSED" "$TIMEOUT"
        [ "$ELAPSED" -ge "$TIMEOUT" ] && { echo; echo "[oneclick] ✗ 超时（调 ONECLICK_TIMEOUT=秒数 重试）"; exit 2; } ;;
  esac
  sleep 10
done
echo
if [[ "$STATUS" != "done" ]]; then
  echo "[oneclick] ✗ 推理失败，服务日志最后 40 行："; tail -40 "$LOG_DIR/serve_oneclick.log"; exit 2
fi
curl -s "http://127.0.0.1:8000/status/$REQ_ID" | python3 -m json.tool | head -20

# ----------------------------- ⑦ 拿回结果 -----------------------------
OUT_DIR=$(curl -s "http://127.0.0.1:8000/status/$REQ_ID" | python3 -c "import json,sys;print(json.load(sys.stdin)['out'])")
echo "[oneclick] ⑦ 结果目录: $OUT_DIR"
N_PRED=$(find "$OUT_DIR" -name prediction.json 2>/dev/null | wc -l)
echo "[oneclick]   prediction.json 数量: $N_PRED / $N_CASES"
[ "$N_PRED" -ge "$N_CASES" ] || { echo "[oneclick] ✗ 答案不全（$N_PRED/$N_CASES）"; ls "$OUT_DIR" | head; exit 2; }
if [ -f "$OUT_DIR/duplicate_pairs.jsonl" ]; then
  echo "[oneclick]   duplicate_pairs.jsonl: $(wc -l < "$OUT_DIR/duplicate_pairs.jsonl") 行，首行："
  head -1 "$OUT_DIR/duplicate_pairs.jsonl"
else
  echo "[oneclick] ✗ 缺 duplicate_pairs.jsonl"; exit 2
fi
FIRST_PRED=$(find "$OUT_DIR" -name prediction.json | sort | head -1)
echo "[oneclick]   答案样例（$FIRST_PRED）："
python3 - "$FIRST_PRED" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(d, ensure_ascii=False, indent=1)[:900])
PY
echo "[oneclick]   掩膜文件数: $(find "$OUT_DIR" -name '*.nii.gz' | wc -l)"

# ----------------------------- ⑧ 回调闭环 -----------------------------
N_CB=$(curl -s http://127.0.0.1:9000/received | python3 -c "import json,sys;print(json.load(sys.stdin)['count'])")
echo "[oneclick] ⑧ mock 平台收到的回调: $N_CB 条"
[ "$N_CB" -ge 1 ] || { echo "[oneclick] ✗ 没收到回调（闭环失败，见 $LOG_DIR/mock_callback.log）"; exit 2; }
curl -s http://127.0.0.1:9000/received | python3 -c "import json,sys
for c in json.load(sys.stdin)['callbacks']:
    print('  ', json.dumps(c['body'], ensure_ascii=False))"

echo "=========================================================================="
echo "[oneclick] ✓ 全流程通过：推理 done → 答案 $N_PRED/$N_CASES → 校验 ok → 回调已收"
echo "[oneclick]   答案目录: $OUT_DIR"
echo "[oneclick]   这就是平台将收到的 predPath —— 可直接在平台发起点评时对照"
echo "=========================================================================="
exit 0
