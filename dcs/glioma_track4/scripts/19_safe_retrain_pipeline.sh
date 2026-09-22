#!/usr/bin/env bash
# ============================================================================
# 无人值守·安全重训流水线："先导出保险，再重训"
#
# 为什么要有"保险"这一步：
#   新划分重训是 14~19h 的长任务，且归档旧产物是不可逆的动作（虽然移动到
#   _local_validation_backup，但旧权重与新划分不兼容、无法复用）。
#   先把现有的折按规范导出到独立目录，重训无论成败，手上始终有一个
#   **完整可提交**的版本 —— 重训从"必须成功的赌博"变成"纯增值"。
#
# 阶段（每一步都会打时间戳日志，便于事后追溯）：
#   1) 等 fold1 续训完成（轮询，最长 WAIT_MAX）
#   2) 导出 3 折保险 → $BACKUP_ROOT/checkpoint/<goal>/（规范 §5.2 布局）
#   3) 校验保险完整性（解析到 3 个权重 + 6 个 goal 目录齐全）
#   4) 停掉对照实验 g4L_fold3（新划分下会重做，且它会占用 GPU）
#   5) 归档旧产物（manifest/folds/缓存/权重/日志，移动到带时间戳的目录）
#   6) 重新探针 + 生成新划分，并**当场校验无重复、无缺失**
#   7) 重建预处理缓存
#   8) 启动 4 折并行训练（第 5 折随后补）
#
# 用法：
#   nohup bash scripts/19_safe_retrain_pipeline.sh > logs/19_pipeline.log 2>&1 &
#   tail -f logs/19_pipeline.log                      # 监控
#
# 可调环境变量：
#   WAIT_MAX=14400     等待 fold1 的最长秒数（默认 4h）
#   BACKUP_ROOT=$HOME/glioma_submission_backup        保险落盘位置
#   NGPU=4             并行折数
#   SKIP_WAIT=1        不等 fold1（确认它已完成后使用）
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python}"
WAIT_MAX="${WAIT_MAX:-14400}"
BACKUP_ROOT="${BACKUP_ROOT:-$HOME/glioma_submission_backup}"
NGPU="${NGPU:-4}"
export WORKSPACE="$BACKUP_ROOT"

log() { printf '[%s][19] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "✗ $*"; exit 1; }

log "==================== 开始（保险 → $BACKUP_ROOT）===================="

# --------------------------------------------------------------------------- #
# 1) 等 fold1 续训完成
# --------------------------------------------------------------------------- #
if [[ "${SKIP_WAIT:-0}" == "1" ]]; then
  log "1/8 SKIP_WAIT=1 → 跳过等待"
else
  log "1/8 等待 fold1 续训完成（最长 ${WAIT_MAX}s）…"
  t=0
  while (( t < WAIT_MAX )); do
    if grep -q "\[trainer\] done" logs/train_fold1_resume.log 2>/dev/null; then
      log "    fold1 已完成 ✔"
      break
    fi
    if ! pgrep -f "training.trainer.*fold 1" > /dev/null 2>&1; then
      die "fold1 进程已消失且日志无 done 标记 —— 需人工检查 logs/train_fold1_resume.log"
    fi
    sleep 60; t=$((t + 60))
    (( t % 600 == 0 )) && log "    等待中… ${t}s（进度：$(grep -c 'epoch ' logs/train_fold1_resume.log 2>/dev/null || echo 0) epoch）"
  done
  (( t < WAIT_MAX )) || die "等待 fold1 超时（${WAIT_MAX}s）"
fi

# --------------------------------------------------------------------------- #
# 2) 导出保险
# --------------------------------------------------------------------------- #
log "2/8 导出 3 折保险 → $BACKUP_ROOT/checkpoint/"
for f in 0 1 2; do
  [[ -f "checkpoints/g4_fold$f/best.pth" ]] || die "缺 checkpoints/g4_fold$f/best.pth，无法导出保险"
done
bash scripts/09_export_submission.sh g4_fold0,g4_fold1,g4_fold2 \
  || die "保险导出失败（不继续归档，避免两头空）"

# --------------------------------------------------------------------------- #
# 3) 校验保险
# --------------------------------------------------------------------------- #
log "3/8 校验保险完整性"
n_w="$(find "$BACKUP_ROOT/checkpoint/goal5_segmentation" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)"
(( n_w >= 3 )) || die "保险里只找到 $n_w 个分割权重（期望 3）"
for g in goal1_authenticity goal2_stitched goal2_duplicate goal3_tumor goal4_diagnosis goal5_segmentation; do
  n="$(find "$BACKUP_ROOT/checkpoint/$g" -maxdepth 1 -type f 2>/dev/null | wc -l)"
  (( n > 0 )) || die "保险缺目录内容：$g"
  log "    · $g: $n 个文件"
done
log "    保险校验通过 ✔（$n_w 个分割权重）"

# --------------------------------------------------------------------------- #
# 4) 停掉对照实验（它会占 GPU，且新划分下要重做）
# --------------------------------------------------------------------------- #
log "4/8 停止 g4L_fold3 对照实验"
pkill -f "train_large" 2>/dev/null || true
sleep 8
log "    已停止"

# --------------------------------------------------------------------------- #
# 5) 归档旧产物
# --------------------------------------------------------------------------- #
log "5/8 归档旧产物（不删除，可回滚）"
bash scripts/17_reset_for_official.sh

# --------------------------------------------------------------------------- #
# 6) 重新探针 + 新划分（并校验）
# --------------------------------------------------------------------------- #
log "6/8 重新探针 + 生成新折划分"
bash scripts/01_probe.sh
bash scripts/02_build_dataset.sh
"$PY" - <<'PYEOF'
import collections, json, os, sys
sys.path.insert(0, os.path.abspath("."))
from src.utils.config import load_paths, resolve
paths = load_paths()
man = json.load(open(resolve(paths["manifest"]), encoding="utf-8"))
folds = json.load(open(resolve(paths["folds"]), encoding="utf-8"))
accs = [c["accession"] for c in man["cases"]]
cnt = collections.Counter(a for k in folds for a in (folds[k].get("val") or []))
dup = sorted(a for a, c in cnt.items() if c > 1)
miss = sorted(a for a in accs if a not in cnt)
print(f"[19] 新划分：{len(folds)} 折 / {len(accs)} 例")
print(f"[19]   各折 val：{ {k: len(folds[k]['val']) for k in sorted(folds)} }")
print(f"[19]   重复：{dup or '无 ✓'}    缺失：{miss or '无 ✓'}")
if dup or miss:
    print("[19] ✗ 新划分不干净，终止（不启动训练）")
    sys.exit(1)
print("[19] ✓ 划分干净：每例恰好属于一个折的 val")
PYEOF

# --------------------------------------------------------------------------- #
# 7) 重建缓存
# --------------------------------------------------------------------------- #
log "7/8 重建预处理缓存"
"$PY" scripts/13_build_cache.py --workers "${CACHE_WORKERS:-8}"

# --------------------------------------------------------------------------- #
# 8) 启动训练
# --------------------------------------------------------------------------- #
log "8/8 启动 $NGPU 折并行训练（新划分）"
bash scripts/03_train.sh all "$NGPU"

cat <<EOF

==============================================================================
[19] 流水线完成，训练已在后台运行。
  · 监控：      tail -f logs/train_fold0.log
  · 第 5 折：   bash scripts/03_train.sh 4
  · 交付数字：  FOLDS="0 1 2 3 4" bash scripts/16_finalize.sh
  · 保险位置：  $BACKUP_ROOT/checkpoint/（可随时用 GLIOMA_CHECKPOINT_ROOT 指回它）
==============================================================================
EOF
