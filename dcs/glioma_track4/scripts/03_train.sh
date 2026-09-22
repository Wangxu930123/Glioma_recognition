#!/usr/bin/env bash
# 训练（默认 5 折；平台容器上限 4 个 → 4 折并行 + 第 5 折后补）
# 用法：
#   bash scripts/03_train.sh 0                     # 单折（前台）
#   bash scripts/03_train.sh all 4                 # 4 折并行（4 张卡）
#   FOLD=2 GPUS="0 1" bash scripts/03_train.sh one
#   PRETRAINED=/path/to.pth bash scripts/03_train.sh 0    # 加载预训练权重（合规性自行确认）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-python}"
TAG_PREFIX="${TAG_PREFIX:-g4}"
CONFIG="${CONFIG:-train}"
mkdir -p logs
PRETRAINED="${PRETRAINED:-}"
EXTRA=()
[[ -n "$PRETRAINED" ]] && EXTRA+=(--pretrained "$PRETRAINED")

MODE="${1:-0}"
if [[ "$MODE" == "all" ]]; then
  NGPU="${2:-4}"
  for f in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$f PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup "$PY" -m src.training.trainer --config "$CONFIG" --fold "$f" --tag "${TAG_PREFIX}_fold$f" \
      "${EXTRA[@]+"${EXTRA[@]}"}" > "logs/train_fold$f.log" 2>&1 &
    echo "[03] fold$f -> GPU$f (logs/train_fold$f.log)"
  done
  echo "[03] 已并行启动 $NGPU 折；第 5 折：bash scripts/03_train.sh 4"
  echo "[03] 监控：tail -f logs/train_fold0.log"
else
  FOLD="${FOLD:-$MODE}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m src.training.trainer --config "$CONFIG" --fold "$FOLD" --tag "${TAG_PREFIX}_fold$FOLD" \
    "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee "logs/train_fold$FOLD.log"
fi
