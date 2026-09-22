#!/usr/bin/env bash
# 在**联网机器**上预下载 wheel，供"禁止联网"的云桌面离线安装
#   bash scripts/00b_prepare_wheels.sh [--with-torch] [--mirror tsinghua] [--out wheels]
# 云桌面内：bash scripts/00_setup_env.sh --mode system --offline --wheels wheels
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT="${OUT:-wheels}"; PYVER="${PYVER:-3.11}"; MIRROR="${MIRROR:-tsinghua}"; WITH_TORCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-torch) WITH_TORCH=1; shift;;
    --mirror) MIRROR="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "未知参数 $1"; exit 2;;
  esac
done
case "$MIRROR" in
  tsinghua) IDX="https://pypi.tuna.tsinghua.edu.cn/simple";;
  aliyun)   IDX="https://mirrors.aliyun.com/pypi/simple";;
  none)     IDX="https://pypi.org/simple";;
esac
mkdir -p "$OUT"
echo "[wheels] 下载非 torch 依赖 → $OUT（目标 python $PYVER）"
grep -v '^[[:space:]]*torch' requirements.txt > /tmp/_req_wheels.txt
python3 -m pip download -r /tmp/_req_wheels.txt -d "$OUT" -i "$IDX"
if [[ $WITH_TORCH == 1 ]]; then
  echo "[wheels] 下载 torch（约 2-3GB，含 nvidia 依赖）"
  python3 -m pip download "torch==2.4.1" -d "$OUT" -i "$IDX"
fi
echo "[wheels] 完成：$(ls -1 "$OUT" | wc -l) 文件，$(du -sh "$OUT" | cut -f1)"
