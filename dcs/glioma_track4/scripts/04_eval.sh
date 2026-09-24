#!/usr/bin/env bash
# 离线评估：Dice/NSD/HD95 + 重复影像 AUC-PR / Recall@10%FPR / Precision@15%Recall
#
# 用法：bash scripts/04_eval.sh [fold] [limit]          例：bash scripts/04_eval.sh 0 30
#       bash scripts/04_eval.sh --split external        官方验证集（全折集成，最终口径）
#       bash scripts/04_eval.sh --split fold 1          强制折内 val（旧口径）
#
# 口径（--split / SPLIT 环境变量，默认 auto）：
#   auto     有 data/manifest_val.json 就评**官方验证集**（全折集成，不做留一），
#            否则评折内 val（多折时按"留一折集成"逐个折跑）
#   fold     折内 val：评估第 f 折时应用其余折模型（留一，避免该折模型见过本折 val）
#   external 官方验证集：验证集与训练集无交集 → 全折集成即无泄漏
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-}"
if [ -z "$PY" ]; then                          # 容器里常常只有 python3
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
    done
fi
[ -n "$PY" ] || { echo "[04] 找不到 python3/python：请 export PY=<解释器路径>" >&2; exit 2; }
SPLIT="${SPLIT:-auto}"
if [[ "${1:-}" == "--split" ]]; then           # 允许 --split external 这种显式写法
    SPLIT="${2:?用法: --split auto|fold|external}"
    shift 2
fi
FOLD="${1:-0}"; LIMIT="${2:-}"
ARGS=(--split "$SPLIT" --fold "$FOLD")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
# CKPT 可显式指定权重（逗号分隔=集成）；不指定时 external 自动取全部折、折内取单折
[[ -n "${CKPT:-}" ]] && ARGS+=(--ckpt "$CKPT")
LOG="logs/eval_fold$FOLD.log"
[[ "$SPLIT" == "external" ]] && LOG="logs/eval_external.log"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m src.evaluation.evaluate "${ARGS[@]}" 2>&1 | tee "$LOG"
