#!/usr/bin/env bash
# ============================================================================
# 按《赛事开发规范》§5.2 导出**提交权重**到平台约定目录：
#
#   /2026aicompetition/workspace/checkpoint/
#   ├── goal1_authenticity/model.pt
#   ├── goal2_stitched/model.pt
#   ├── goal2_duplicate/encoder.pt
#   ├── goal3_tumor/model.pt
#   ├── goal4_diagnosis/model.pt
#   └── goal5_segmentation/<tag>.pt        ← 多折时保留全部，供集成
#
# 本工程是**一个多任务骨干**（一次前向同时产出分割/结构化/特殊影像/嵌入），
# 因此同一份权重会导出到各 goal 目录（默认硬链接，不额外占磁盘）。
#
# 用法：
#   bash scripts/09_export_submission.sh                 # 全部 g4_fold* 做集成
#   bash scripts/09_export_submission.sh g4_fold0,g4_fold1
#   bash scripts/09_export_submission.sh --verify        # 只复核已导出的路径
#   WORKSPACE=/path/to/workspace bash scripts/09_export_submission.sh
#
# 注意：**不要**用 smoke_fold* / _bench.pth 之类的临时产物做提交权重。
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python}"
WS="${WORKSPACE:-/2026aicompetition/workspace}"
export GLIOMA_CHECKPOINT_ROOT="${GLIOMA_CHECKPOINT_ROOT:-$WS/checkpoint}"

if [[ "${1:-}" == "--verify" ]]; then
  echo "[09] 复核已导出权重：$GLIOMA_CHECKPOINT_ROOT"
  exec "$PY" -m integration.export_ckpt --verify
fi

FOLDS="${1:-}"
if [[ -z "$FOLDS" ]]; then
  # 排除 smoke/bench 等临时折，只取正式训练折
  FOLDS="$(ls -d checkpoints/g4_fold*/ 2>/dev/null | xargs -r -n1 basename | paste -sd, -)"
  if [[ -z "$FOLDS" ]]; then
    # 只训了**全量模型**时（`03_train.sh full` → checkpoints/g4_full/）默认导出它：
    # 否则这里报"请先训练"，而磁盘上明明有训好的全量权重 —— 演练时极易误判成训练失败。
    FOLDS="$(ls -d checkpoints/g4_full*/ 2>/dev/null | xargs -r -n1 basename | paste -sd, -)"
    [[ -n "$FOLDS" ]] && echo "[09] 未找到折权重，改用全量权重：$FOLDS"
  fi
  if [[ -z "$FOLDS" ]]; then
    echo "[09] ✗ 未找到 checkpoints/g4_fold*/best.pth 或 checkpoints/g4_full*/best.pth，请先训练"
    exit 2
  fi
  echo "[09] 自动选择正式权重：$FOLDS"
fi

# 逐折确认权重存在，缺折直接失败（避免导出半套权重）
IFS=',' read -ra arr <<< "$FOLDS"
NORM=()                                   # 归一化：折号 0 → tag g4_fold0；已是 tag 则原样
for tag in "${arr[@]}"; do
  tag="${tag// /}"
  [[ -z "$tag" ]] && continue
  case "$tag" in
    *fold*) ;;                            # 已含 fold（g4_fold0 / g4L_fold3）→ 原样
    full) tag="g4_full" ;;                # 全量训练产物固定叫 g4_full
    g4_*|g4L_*) ;;                        # 已是完整 tag（g4_full / g4_full43）→ 原样
    *) tag="g4_fold$tag" ;;               # 只给了折号 → 补前缀
  esac
  [[ -f "checkpoints/$tag/best.pth" ]] || { echo "[09] ✗ 缺少 checkpoints/$tag/best.pth"; exit 2; }
  NORM+=("$tag")
done
[[ ${#NORM[@]} -gt 0 ]] || { echo "[09] ✗ 未解析到任何有效折"; exit 2; }
FOLDS="$(IFS=','; echo "${NORM[*]}")"
echo "[09] 归一化后的折：$FOLDS"

# ⚠️ 演练权重 vs 提交权重
# `checkpoint/` 是**平台评分真正读取的路径**。用 1~2 个 epoch 的冒烟权重演练完
# 链路后若忘了换回正式权重，提交的就是"看起来齐全、实际未训练"的模型 ——
# 而 `--verify` 只检查文件是否齐备、不会发现这件事。
case "$FOLDS" in
  *smoke*|*bench*|*debug*|*demo*|*rehearsal*)
    echo
    echo "  ⚠️  导出的折名含演练标记（$FOLDS）。"
    echo "      若这是为了走通链路，请导出到**独立目录**，避免污染提交路径："
    echo "        WORKSPACE=/tmp/ws_rehearsal bash scripts/09_export_submission.sh $FOLDS"
    echo "      演练结束后，务必用正式折重新导出到 $GLIOMA_CHECKPOINT_ROOT ："
    echo "        bash scripts/09_export_submission.sh"
    echo
    ;;
esac

echo "[09] 目标：$GLIOMA_CHECKPOINT_ROOT"
# 正式提交用 copy（硬链接在跨文件系统/打包上传时不保证成立）
"$PY" -m integration.export_ckpt --mode copy --folds "$FOLDS"

echo
echo "[09] 复核导出结果 …"
"$PY" -m integration.export_ckpt --verify

cat <<EOF

------------------------------------------------------------------------
[09] 完成。后续步骤：
  1) 多折集成时，测评容器里把全部折用逗号传入：
       GLIOMA_CKPT=$GLIOMA_CHECKPOINT_ROOT/goal5_segmentation/g4_fold0.pt,$GLIOMA_CHECKPOINT_ROOT/goal5_segmentation/g4_fold1.pt,...
     （不设置时，common.resolve_checkpoints() 会自动读取该目录下全部 .pt 做集成）
  2) 跑一次 Mock Competition 验证闭环：
       bash scripts/08_mock_competition.sh
  3) 跑提交前检查清单：
       bash scripts/23_pre_submit_check.sh
------------------------------------------------------------------------
EOF
