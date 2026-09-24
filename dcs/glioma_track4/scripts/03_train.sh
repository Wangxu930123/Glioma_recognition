#!/usr/bin/env bash
# 训练（默认 5 折；平台容器上限 4 个 → 4 折并行 + 第 5 折后补）
# 用法：
#   bash scripts/03_train.sh 0                     # 单折（前台）
#   bash scripts/03_train.sh all 4                 # 4 折并行（4 张卡）
#   bash scripts/03_train.sh full                  # **全量训练**（train=全部病例，
#                                                  #   val=官方验证集 → checkpoints/g4_full/）
#   PRETRAINED=/path/to.pth bash scripts/03_train.sh 0    # 加载预训练权重（合规性自行确认）
#   CONFIG=train_large bash scripts/03_train.sh 0         # 换配置（默认 train）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-}"
if [ -z "$PY" ]; then                          # 容器里常常只有 python3
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
    done
fi
[ -n "$PY" ] || { echo "[03] 找不到 python3/python：请 export PY=<解释器路径>" >&2; exit 2; }
TAG_PREFIX="${TAG_PREFIX:-g4}"
CONFIG="${CONFIG:-train}"
mkdir -p logs
PRETRAINED="${PRETRAINED:-}"
EXTRA=()
[[ -n "$PRETRAINED" ]] && EXTRA+=(--pretrained "$PRETRAINED")

MODE="${1:-0}"
if [[ "$MODE" == "full" ]]; then
  # 全量训练：不做交叉验证，train = 清单全部病例，val = 官方验证集
  #   （需要 data/manifest_val.json：export VAL_ROOT=... && bash scripts/01_probe.sh --val）
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m src.training.trainer --config "$CONFIG" --fold full --tag "${TAG_PREFIX}_full" \
    "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee "logs/train_full.log"
elif [[ "$MODE" == "all" ]]; then
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
