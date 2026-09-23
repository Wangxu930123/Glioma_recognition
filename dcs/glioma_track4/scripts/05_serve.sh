#!/usr/bin/env bash
# 本地起服务并自测（/health + 规范 /call 嵌套体 + 扁平兼容），无需真实数据
# 用法：CKPT=checkpoints/g4_fold0/best.pth bash scripts/05_serve.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-}"
if [ -z "$PY" ]; then                          # 容器里常常只有 python3
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
    done
fi
[ -n "$PY" ] || { echo "[05] 找不到 python3/python：请 export PY=<解释器路径>" >&2; exit 2; }
PORT="${PORT:-8000}"
CKPT="${CKPT:-$(ls -1 checkpoints/g4_fold*/best.pth 2>/dev/null | head -1 || true)}"

if [[ -z "$CKPT" ]]; then
  echo "[05] 未找到权重：先训练（bash scripts/03_train.sh 0）或指定 CKPT=..."
  exit 1
fi
export GLIOMA_CKPT="$CKPT"
mkdir -p logs
echo "[05] 使用权重: $CKPT"
nohup "$PY" -m src.serving.app --port "$PORT" > logs/serving.out 2>&1 &
echo $! > logs/serving.pid
sleep 6

echo "--- /health ---"
curl -s -m 5 "http://127.0.0.1:$PORT/health" && echo
echo "--- 规范 /call（嵌套 input）---"
DS="${DS:-/tmp/glioma_eval_ds}"
mkdir -p "$DS"
curl -s -m 10 -X POST "http://127.0.0.1:$PORT/call" -H 'Content-Type: application/json' \
  -d "{\"request_id\":\"test-1\",\"team_id\":\"t\",\"track_code\":\"4\",\"input\":{\"evaluation_id\":\"eval_test\",\"dataset_path\":\"$DS\"}}" && echo
echo "--- 停止服务 ---"
kill "$(cat logs/serving.pid)" 2>/dev/null || true
echo "[05] 完成（日志 logs/serving.out）"
