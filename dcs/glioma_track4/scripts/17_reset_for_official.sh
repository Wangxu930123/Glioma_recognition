#!/usr/bin/env bash
# ============================================================================
# 进入「正式数据训练」前的清理：把本地/公开数据（如 BraTS）的验证产物归档移出，
# 避免与官方数据的产物混用，也避免不规范的数据源记录进入提交日志。
# ----------------------------------------------------------------------------
# 为什么必须做：
#   · 赛事规则要求"比赛数据只能在大赛专属环境中使用"，且日志需可追溯数据来源；
#   · 本工程的 `assert_data_source` 闸门会拒绝"清单数据源与本机数据源不同类"的训练，
#     因此切到官方数据前必须重新生成清单；
#   · 提交物中的 `logs/*.jsonl` 应为**官方数据训练**的日志。
#
# 本脚本**不删除**任何东西：把本地验证产物移动到 `_local_validation_backup/`，
# 可随时恢复。用法：
#   bash scripts/17_reset_for_official.sh
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
BK="$ROOT/_local_validation_backup/$TS"
mkdir -p "$BK"

echo "[17] 归档本地验证产物 -> $BK"

move() {  # move <path> <dest-name>
  local src="$1" name="$2"
  if [[ -e "$src" ]]; then
    mv "$src" "$BK/$name"
    echo "  · $src  ->  _local_validation_backup/$TS/$name"
  fi
}

# 1) 数据清单与折划分（含 data_source 标识，必须按官方数据重新生成）
move data/manifest.json manifest.json
move data/folds.json folds.json
# 2) 预处理缓存（体素级数据，体积极大，且来自本地数据）
move data/preprocess_cache preprocess_cache
# 3) 本地验证的模型权重
if compgen -G "checkpoints/g4_fold*" > /dev/null; then
  mkdir -p "$BK/checkpoints"
  for d in checkpoints/g4_fold*; do mv "$d" "$BK/checkpoints/"; done
  echo "  · checkpoints/g4_fold*  ->  _local_validation_backup/$TS/checkpoints/"
fi
move checkpoints/smoke_fold0 smoke_fold0
# 4) 本地验证日志（提交日志必须来自官方数据训练）
if compgen -G "logs/*.jsonl" > /dev/null; then
  mkdir -p "$BK/logs"
  mv logs/*.jsonl "$BK/logs/" 2>/dev/null || true
  echo "  · logs/*.jsonl  ->  _local_validation_backup/$TS/logs/"
fi

echo
echo "[17] 完成。接下来用**官方数据**重建："
echo "    export DATASET_ROOT=/2026aicompetition/datasets/training        # 官方训练集根（影像在 training/annotation/，会自动下钻）"
echo "    export CACHE_DIR=/2026aicompetition/workspace/cache             # 缓存放私有存储（容器删除不丢）"
echo "    bash scripts/01_probe.sh                                        # 探针（会写入 data_source=official/train_v1）"
echo "    bash scripts/02_build_dataset.sh                                # 分层折划分"
echo "    python scripts/13_build_cache.py --workers 8                     # 预处理缓存"
echo "    bash scripts/03_train.sh all 4                                   # 4 折并行训练"
echo
echo "    提示：本地/公开数据的验证结果请只用于工程正确性判断，不要写入技术报告作为比赛成绩。"
