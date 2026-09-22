#!/usr/bin/env bash
# ============================================================================
# 训推平台「测评容器」启动命令（把本行填进容器实例的"启动命令"）
# ----------------------------------------------------------------------------
#   CALLBACK_URL="http://<平台地址>/api/competition/inference/callback/" \
#     bash /2026aicompetition/workspace/glioma_track4/scripts/06_platform_serve.sh
#
# 平台硬性要求（《赛事开发规范》）：
#   · 8000 端口 + /health 必须常活（启动探针 5s/次、最长 1800s；探活 30s/次）；
#   · 启动命令必须是**长期运行的前台进程**；
#   · /call 必须 5s 内回 200（本工程为后台异步）。
#
# ⚠️ 回调地址：容器实例页面上方会显示完整回调地址，**务必通过 CALLBACK_URL 传入**，
#    否则推理完成后无法通知平台闭环（该次测评不计分）。
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

WS="${WORKSPACE:-/2026aicompetition/workspace}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOG_DIR="$WS/logs"; mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LOGS_DIR="$LOG_DIR"
export MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-2}"

# ★ 正式评测开启“加载容错”：单个序列/文件异常时跳过并继续，而不是让**整批**测评失败。
#   赛事评测不可重跑，测试集里任意 1 个脏文件若触发 fail-fast 会导致**全部病例 0 分**；
#   开启后最坏情况只损失该例。默认关闭以保持团队既有契约（tests/test_streaming.py 有断言），
#   仅在正式评测（本脚本）中开启。
export GLIOMA_LOADER_TOLERANT="${GLIOMA_LOADER_TOLERANT:-1}"

echo "[06] repo=$ROOT  workspace=$WS  port=$PORT"

# ---- 权重自动发现（私有存储优先；多折用逗号拼接做集成）----
find_ckpts() {
  local hits
  hits="$(ls -1 $WS/train_model/g4_fold*/best.pth 2>/dev/null | sort | paste -sd, -)"
  [[ -z "$hits" ]] && hits="$(ls -1 $ROOT/checkpoints/g4_fold*/best.pth 2>/dev/null | sort | paste -sd, -)"
  echo "$hits"
}
if [[ -z "${GLIOMA_CKPT:-}" ]]; then
  GLIOMA_CKPT="$(find_ckpts)"
fi
if [[ -n "${GLIOMA_CKPT:-}" ]]; then
  export GLIOMA_CKPT
  echo "[06] 推理权重: $GLIOMA_CKPT"
else
  echo "[06] ⚠️ 未找到权重（$WS/train_model/g4_fold*/best.pth 或 $ROOT/checkpoints/…）"
  echo "[06]    服务仍会启动以满足平台 /health 探测，但 /call 会失败——请先训练并导出权重"
fi

# ---- 回调地址：多来源解析（缺省时给出醒目告警，避免"静默不回调"导致不计分）----
# 为什么不静默：推理服务必须回调平台才能闭环，缺少地址 = 该次测评不计分。
if [[ -z "${CALLBACK_URL:-}" ]]; then
  for k in PLATFORM_CALLBACK_URL COMPETITION_CALLBACK_URL INFERENCE_CALLBACK_URL CALLBACK_ADDR CALLBACK; do
    if [[ -n "${!k:-}" ]]; then export CALLBACK_URL="${!k}"; break; fi
  done
fi
if [[ -z "${CALLBACK_URL:-}" ]]; then
  for f in "$WS/callback_url.txt" "$ROOT/callback_url.txt" "$ROOT/configs/callback_url.txt"; do
    if [[ -s "$f" ]]; then export CALLBACK_URL="$(tr -d '\r\n' < "$f")"; break; fi
  done
fi
if [[ -n "${CALLBACK_URL:-}" ]]; then
  echo "[06] ✓ 回调地址: $CALLBACK_URL"
else
  echo "=============================================================================="
  echo "[06] ⚠️⚠️ 未解析到回调地址：推理完成后**无法回调平台，本次测评可能不计分**！"
  echo "[06]      任选其一："
  echo "[06]        1) 启动命令传入（推荐）："
  echo "[06]           CALLBACK_URL='http://<平台>/api/competition/inference/callback/' bash $0"
  echo "[06]        2) 写入文件后再启动： echo 'http://.../callback/' > $WS/callback_url.txt"
  echo "[06]      完整回调地址见「容器实例页面」上方。服务仍会启动（/health 正常），但不闭环。"
  echo "=============================================================================="
fi
[[ -n "${DICOM_CACHE:-}" ]] && echo "[06] DICOM 缓存目录: $DICOM_CACHE"

# ---- 依赖与 GPU 自检（仅提示）----
python3 - <<'PY' || echo "[06] ⚠️ 依赖自检未过：先跑 bash scripts/00_setup_env.sh --mode system"
import importlib, sys
miss = [m for m in ("torch", "numpy", "nibabel", "SimpleITK", "skimage", "fastapi", "uvicorn")
        if importlib.util.find_spec(m) is None]
import torch
print(f"[06] torch {torch.__version__} cuda {torch.version.cuda} 可用 {torch.cuda.is_available()}")
print(f"[06] 缺失依赖: {miss if miss else '无 ✓'}")
sys.exit(1 if miss else 0)
PY

# ---- 前台常驻（容器生命期 = 服务生命期）----
echo "[06] 启动服务 $HOST:$PORT（日志: $LOG_DIR/serving.out）"
exec python3 -m src.serving.app --host "$HOST" --port "$PORT" 2>&1 | tee -a "$LOG_DIR/serving.out"
