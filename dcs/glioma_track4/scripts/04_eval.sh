#!/usr/bin/env bash
# 离线评估：折内 val 的 Dice/NSD/HD95 + 重复影像 AUC-PR / Recall@10%FPR / Precision@15%Recall
# 用法：bash scripts/04_eval.sh [fold] [limit]     例如 bash scripts/04_eval.sh 0 30
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-}"
if [ -z "$PY" ]; then                          # 容器里常常只有 python3
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
    done
fi
[ -n "$PY" ] || { echo "[04] 找不到 python3/python：请 export PY=<解释器路径>" >&2; exit 2; }
FOLD="${1:-0}"; LIMIT="${2:-}"
ARGS=(--fold "$FOLD")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m src.evaluation.evaluate "${ARGS[@]}" 2>&1 | tee "logs/eval_fold$FOLD.log"
