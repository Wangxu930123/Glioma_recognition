#!/usr/bin/env bash
# ============================================================================
# 用**修复后的折划分**从零重训（5 折），拿到"无划分缺陷 + 多折集成"的干净数字。
#
# 为什么必须从头重训（重要）：
#   旧权重的训练集基于**旧划分**，而新旧划分是随机独立的、val 会重叠约 80%。
#   若拿旧权重按新划分评估 → 模型评到自己训练过的病例 → 严重数据泄漏。
#   因此旧权重只能配旧划分使用，二者不可混搭。
#
# 本脚本做六件事（全部幂等、不删除任何东西）：
#   1) 停止当前训练进程
#   2) 归档旧产物（manifest / folds / 缓存 / 权重 / 日志）到 _local_validation_backup/<时间戳>/
#   3) 重新探针（生成新的 manifest）
#   4) 生成新折划分，并**当场校验**无重复、无缺失
#   5) 重建预处理缓存（体素级，耗时较长）
#   6) 启动 4 折并行训练（第 5 折随后补：bash scripts/03_train.sh 4）
#
# 用法：
#   bash scripts/18_retrain_new_folds.sh                # 完整流程
#   SKIP_ARCHIVE=1 bash scripts/18_retrain_new_folds.sh # 已归档过 → 只重划分+训练
#   SKIP_CACHE=1   bash scripts/18_retrain_new_folds.sh # 缓存可复用 → 跳过第 5 步
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python}"
NGPU="${NGPU:-4}"

echo "==================== 1/6 停止当前训练 ===================="
if pgrep -f "src.training.trainer" > /dev/null 2>&1; then
  pkill -f "src.training.trainer" || true
  sleep 6
  echo "[18] 已停止训练进程"
else
  echo "[18] 当前没有训练进程"
fi

echo
echo "==================== 2/6 归档旧产物（不删除，可回滚）===================="
if [[ "${SKIP_ARCHIVE:-0}" == "1" ]]; then
  echo "[18] SKIP_ARCHIVE=1 → 跳过归档"
else
  bash scripts/17_reset_for_official.sh
fi

echo
echo "==================== 3/6 重新探针 ===================="
if [[ "${SKIP_ARCHIVE:-0}" == "1" && -f data/manifest.json ]]; then
  echo "[18] 清单已存在 → 跳过（如需重建请先删除 data/manifest.json）"
else
  bash scripts/01_probe.sh
fi

echo
echo "==================== 4/6 生成新折划分并校验 ===================="
bash scripts/02_build_dataset.sh
# 校验（注意：build_folds 现在默认**复用**已有划分，此处刚生成的即新划分）
"$PY" - <<'PYEOF'
import collections
import json
import os
import sys
sys.path.insert(0, os.path.abspath("."))
from src.utils.config import load_paths, resolve

paths = load_paths()
man = json.load(open(resolve(paths["manifest"]), encoding="utf-8"))
folds = json.load(open(resolve(paths["folds"]), encoding="utf-8"))
accs = [c["accession"] for c in man["cases"]]
cnt = collections.Counter(a for k in folds for a in (folds[k].get("val") or []))
dup = sorted(a for a, c in cnt.items() if c > 1)
miss = sorted(a for a in accs if a not in cnt)
print(f"[18] 新划分：{len(folds)} 折，病例 {len(accs)}")
print(f"[18]   各折 val 规模：{ {k: len(folds[k]['val']) for k in sorted(folds)} }")
print(f"[18]   重复病例：{dup or '无 ✓'}")
print(f"[18]   缺失病例：{miss or '无 ✓'}")
if dup or miss:
    print("[18] ✗ 新划分仍不干净，请检查 build_folds 实现")
    sys.exit(1)
print("[18] ✓ 划分干净：每例恰好属于一个折的 val（OOF 可覆盖全集）")
PYEOF

echo
echo "==================== 5/6 重建预处理缓存 ===================="
if [[ "${SKIP_CACHE:-0}" == "1" ]]; then
  echo "[18] SKIP_CACHE=1 → 跳过缓存重建"
else
  "$PY" scripts/13_build_cache.py --workers "${CACHE_WORKERS:-8}"
fi

echo
echo "==================== 6/6 启动 4 折并行训练 ===================="
bash scripts/03_train.sh all "$NGPU"

cat <<EOF

------------------------------------------------------------------------
[18] 训练已在后台启动。建议随后执行：
  ① 监控：      tail -f logs/train_fold0.log
  ② 第 5 折：   bash scripts/03_train.sh 4          # 前 4 折收敛后补跑
  ③ 交付数字：  FOLDS="0 1 2 3 4" bash scripts/16_finalize.sh
                （集成阈值标定 → 留一折集成全图评估 → OOF 目标一二/重复
                  → 规范导出权重 → Mock Competition → 提交前检查清单）

把本机时间预估（以实测 208s/epoch、无争用为基准）：
  · 单折 100 epoch ≈ 5.8h；4 折并行受显存/带宽争用影响约 8~11h
  · 第 5 折另需 ≈ 6~8h  →  5 折齐全合计约 14~19h
------------------------------------------------------------------------
EOF
