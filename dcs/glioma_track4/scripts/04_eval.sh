#!/usr/bin/env bash
# 离线评估：折内 val 的 Dice/NSD/HD95 + 重复影像 AUC-PR / Recall@10%FPR / Precision@15%Recall
# 用法：bash scripts/04_eval.sh [fold] [limit]     例如 bash scripts/04_eval.sh 0 30
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-python}"
FOLD="${1:-0}"; LIMIT="${2:-}"
ARGS=(--fold "$FOLD")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m src.evaluation.evaluate "${ARGS[@]}" 2>&1 | tee "logs/eval_fold$FOLD.log"
