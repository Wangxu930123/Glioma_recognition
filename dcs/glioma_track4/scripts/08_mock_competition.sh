#!/usr/bin/env bash
# ============================================================================
# Mock Competition：按平台真实协议跑**完整闭环**，验证"无缝对接"。
#
# 覆盖平台的硬性要求：
#   · /health 常活（启动探针 5s/次、探活 30s/次）        → PASS Health
#   · /call 必须 5s 内返回 200（本工程为后台异步）        → PASS Call response time
#   · 推理在后台执行，完成后回调平台                       → PASS Background / Callback
#   · 答案（prediction.json + 掩码）结构与几何合规          → PASS Output and NIfTI validation
#
# 用法：
#   bash scripts/08_mock_competition.sh
#   bash scripts/08_mock_competition.sh g4_fold0,g4_fold1     # 指定折（多折=集成）
#   GLIOMA_CKPT=/abs/a.pth,/abs/b.pth bash scripts/08_mock_competition.sh
#
# 说明：Mock 用的是**合成小数据集**，目的是验证协议闭环而非精度，
#       因此默认只加载**最快的一折**（多折集成会慢 N 倍，留到全图评估时用）。
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
TEAM="${TEAM_ROOT:-../Glioma_recognition-main}"
PY="${PYTHON:-python}"

# ---- 1) 解析权重：显式 GLIOMA_CKPT 优先，否则取指定的折 / 第一折 ----
FOLDS="${1:-}"
if [[ -n "${GLIOMA_CKPT:-}" ]]; then
  CKPT="$GLIOMA_CKPT"
  echo "[mock] 权重（GLIOMA_CKPT）: $CKPT"
elif [[ -n "$FOLDS" ]]; then
  CKPT=""
  IFS=',' read -ra arr <<< "$FOLDS"
  for tag in "${arr[@]}"; do
    p="checkpoints/${tag}/best.pth"
    [[ -f "$p" ]] && CKPT="${CKPT:+$CKPT,}$(realpath "$p")" || echo "[mock] ⚠️ 缺少 $p"
  done
  echo "[mock] 权重（指定折 $FOLDS）: $CKPT"
else
  first="$(ls -d checkpoints/g4_fold*/best.pth 2>/dev/null | head -1 || true)"
  if [[ -z "$first" ]]; then
    echo "[mock] ✗ 找不到 checkpoints/g4_fold*/best.pth；请先训练或用 GLIOMA_CKPT 指定"
    exit 2
  fi
  CKPT="$(realpath "$first")"
  echo "[mock] 权重（自动取第一折）: $CKPT"
fi
[[ -f "${CKPT%%,*}" ]] || { echo "[mock] ✗ 权重不存在: ${CKPT%%,*}"; exit 2; }

if [[ ! -f "$TEAM/scripts/mock_competition.py" ]]; then
  echo "[mock] ✗ 找不到团队仓库（TEAM_ROOT=$TEAM）"
  exit 2
fi

# ---- 2) 环境变量（与正式启动脚本 06 保持一致）----
export COMPETITION_PIPELINE_FACTORY="tasks.glioma.pipeline:build_pipeline"
export GLIOMA_CKPT="$CKPT"
export GLIOMA_LOADER_TOLERANT="${GLIOMA_LOADER_TOLERANT:-1}"   # 正式评测同款容错
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[mock] 工厂=$COMPETITION_PIPELINE_FACTORY"
echo "[mock] 容错=$GLIOMA_LOADER_TOLERANT"
echo "[mock] 启动团队 Mock Competition …"
echo "------------------------------------------------------------------------"
cd "$TEAM"
exec "$PY" scripts/mock_competition.py --timeout "${MOCK_TIMEOUT:-600}"
