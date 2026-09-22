#!/usr/bin/env bash
# 把训练产物/日志导出到平台**持久化目录**（容器删除不丢），供测评容器复用。
#   bash scripts/07_export_weights.sh            # 导出 checkpoints/ + logs/ + data/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
WS="${WORKSPACE:-/2026aicompetition/workspace}"

mkdir -p "$WS/train_model" "$WS/logs"
if compgen -G "checkpoints/g4_fold*/best.pth" > /dev/null; then
  cp -v checkpoints/g4_fold*/best.pth "$WS/train_model/" 2>/dev/null || true
  for d in checkpoints/g4_fold*/; do
    tag="$(basename "$d")"
    mkdir -p "$WS/train_model/$tag"
    cp -f "$d"/best.pth "$WS/train_model/$tag/" 2>/dev/null || true
  done
  echo "[07] 权重 -> $WS/train_model/"
else
  echo "[07] ⚠️ 未找到 checkpoints/g4_fold*/best.pth，先训练（bash scripts/03_train.sh 0）"
fi

cp -f logs/*.jsonl "$WS/logs/" 2>/dev/null || true
for f in data/manifest.json data/folds.json; do
  [[ -f "$f" ]] && cp -f "$f" "$WS/" 2>/dev/null || true
done
echo "[07] 日志 -> $WS/logs/  ；清单 -> $WS/"
echo "[07] 完成。测评容器启动命令："
echo "     CALLBACK_URL=<平台回调地址> bash $WS/glioma_track4/scripts/06_platform_serve.sh"
