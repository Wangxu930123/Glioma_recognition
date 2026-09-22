#!/usr/bin/env bash
# ============================================================================
# 赛道4 · 一键安装依赖 + 环境自检
# ----------------------------------------------------------------------------
# 场景（按需选一）：
#   本地/云主机（有网）      : bash scripts/00_setup_env.sh
#   训推平台容器（镜像带 torch）: bash scripts/00_setup_env.sh --mode system
#   云桌面（禁止联网）        : bash scripts/00_setup_env.sh --mode system --offline --wheels wheels
#   只体检不安装             : bash scripts/00_setup_env.sh --mode verify
# 其他：--mirror tsinghua | --torch-index mirror | --name <env> | --python <解释器> | --with-torch
# ============================================================================
set -euo pipefail

MODE=fresh; NAME=glioma4; BASE_ENV=""; PYTHON_BIN=""; MIRROR=auto; TORCHSRC=official
OFFLINE=0; WHEELS="${WHEELS_DIR:-wheels}"; VERIFY=1; PYVER="${PYVER:-3.11}"
REQ="requirements.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE="$2"; shift 2;;
    --name)        NAME="$2"; shift 2;;
    --base)        BASE_ENV="$2"; shift 2;;
    --python)      PYTHON_BIN="$2"; shift 2;;
    --mirror)      MIRROR="$2"; shift 2;;
    --torch-index) TORCHSRC="$2"; shift 2;;
    --offline)     OFFLINE=1; shift;;
    --wheels)      OFFLINE=1; WHEELS="$2"; shift 2;;
    --with-torch)  WITH_TORCH=1; shift;;
    --no-verify)   VERIFY=0; shift;;
    -h|--help)     sed -n '2,16p' "$0"; exit 0;;
    *) echo "[setup] 未知参数: $1"; exit 2;;
  esac
done
WITH_TORCH="${WITH_TORCH:-0}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[[ -f requirements-lock.txt ]] && REQ="requirements-lock.txt"

case "$MIRROR" in
  tsinghua) IDX="https://pypi.tuna.tsinghua.edu.cn/simple";;
  aliyun)   IDX="https://mirrors.aliyun.com/pypi/simple";;
  none)     IDX="";;
  *)        IDX="";;
esac

pip_do() {
  local PY="$1"; shift
  local args=(); [[ -n "$IDX" ]] && args=(-i "$IDX")
  if [[ "$OFFLINE" == 1 ]]; then
    [[ -d "$WHEELS" ]] || { echo "[setup] ✗ 离线模式需本地 wheel 目录（先在联网机器跑 scripts/00b_prepare_wheels.sh）: $WHEELS" >&2; return 1; }
    args=(--no-index --find-links "$WHEELS")
  fi
  "$PY" -m pip install "$@" "${args[@]+"${args[@]}"}"
}

echo "==================== 1/3 准备 Python 环境 ===================="
if [[ -n "$PYTHON_BIN" ]]; then
  PY="$PYTHON_BIN"; MODE=verify
elif [[ "$MODE" == "system" ]]; then
  PY="${PYTHON:-}"
  if [[ -z "$PY" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then PY="$CONDA_PREFIX/bin/python"
    else PY="$(command -v python3)"; fi
  fi
elif [[ "$MODE" == "reuse" ]]; then
  "$BASE_ENV/bin/python" -m venv --system-site-packages .venv
  PY="$ROOT/.venv/bin/python"
elif [[ "$MODE" == "verify" ]]; then
  PY="$(command -v python3)"
else
  if command -v conda >/dev/null 2>&1; then
    conda env list | awk '{print $1}' | grep -qx "$NAME" || conda create -y -n "$NAME" "python=$PYVER"
    PY="$(conda info --base)/envs/$NAME/bin/python"
  else
    python3 -m venv .venv; PY="$ROOT/.venv/bin/python"
  fi
fi
[[ -x "$PY" ]] || { echo "[setup] 找不到解释器 $PY"; exit 2; }
echo "[setup] python: $PY ($("$PY" -V 2>&1))"
[[ "$MODE" != "verify" ]] && { "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true; }

if [[ "$MODE" != "verify" && "$MODE" != "system" ]]; then
  echo "==================== 2/3 安装 torch ===================="
  if "$PY" -c "import torch" 2>/dev/null; then
    echo "[setup] 已存在 torch $("$PY" -c 'import torch;print(torch.__version__)')，跳过"
  elif [[ "$TORCHSRC" == "mirror" ]]; then
    echo "[setup] 从 PyPI 镜像装 torch（约 2-3GB，含 nvidia 依赖）"
    pip_do "$PY" "torch==2.4.1"
  else
    CUDA="${CUDA:-124}"
    echo "[setup] 从 download.pytorch.org/whl/cu$CUDA 装 torch"
    "$PY" -m pip install "torch==2.4.1+cu$CUDA" --index-url "https://download.pytorch.org/whl/cu$CUDA" || \
      "$PY" -m pip install "torch>=2.4" --index-url "https://download.pytorch.org/whl/cu$CUDA"
  fi
elif [[ "$MODE" == "system" ]]; then
  echo "==================== 2/3 沿用镜像自带 torch ===================="
  "$PY" -c "import torch;print('[setup] torch',torch.__version__,'cuda',torch.version.cuda,'可用',torch.cuda.is_available())" || {
    echo "[setup] ✗ 当前环境无 torch：平台容器应自带；否则用默认 fresh 模式"; exit 1; }
fi

if [[ "$MODE" != "verify" ]]; then
  echo "==================== 3/3 安装其余依赖（$REQ） ===================="
  grep -v '^[[:space:]]*torch' "$REQ" > /tmp/_req.txt
  if [[ "$MODE" == "system" ]]; then
    pip_do "$PY" --upgrade-strategy only-if-needed -r /tmp/_req.txt   # 不动镜像已有版本
  else
    pip_do "$PY" -r /tmp/_req.txt
  fi
fi

echo "==================== 自检 ===================="
"$PY" - <<'PY'
import importlib, sys
miss = []
for m in ("torch", "numpy", "scipy", "nibabel", "SimpleITK", "skimage", "pandas", "yaml", "fastapi", "uvicorn", "requests"):
    try: importlib.import_module(m)
    except Exception as e: miss.append(f"{m}: {str(e)[:50]}")
import torch
print(f"[check] python {sys.version.split()[0]} | torch {torch.__version__} | cuda {torch.version.cuda} | 可用 {torch.cuda.is_available()}")
print(f"[check] 缺失依赖: {miss if miss else '无 ✓'}")
sys.exit(1 if miss else 0)
PY

if [[ "$VERIFY" == 1 ]]; then
  echo "[setup] 跑合成自检（无需真实数据）..."
  "$PY" scripts/99_smoke_test.py 2>&1 | tail -12
fi
echo "[setup] 完成。下一步：bash scripts/01_probe.sh（实测数据目录结构）"
