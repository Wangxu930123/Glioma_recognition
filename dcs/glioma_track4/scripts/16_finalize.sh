#!/usr/bin/env bash
# ============================================================================
# 训练完成后的收尾流水线（一键出"接近提交状态的数字" + 出提交物）
#
#   FOLDS="0 1 2" bash scripts/16_finalize.sh
#   FOLDS="0 1 2 3" SKIP_MOCK=1 bash scripts/16_finalize.sh
#
# 评估口径（是否接官方验证集，取决于是不是配了 VAL_ROOT / raw.val 并跑过
#   `bash scripts/01_probe.sh --val`）：
#   · 有验证集：**最终指标以官方验证集为准** —— 全折集成、不做留一
#     （验证集与训练集无交集，留一反而每条都少用一个模型）
#   · 没有验证集：原样回退折内 val（1/5 折内标定、2/5 留一折集成、3/5 OOF）
#
# 步骤：
#   1/5 集成阈值标定   —— 用**集成模型**在全图上扫出最优阈值并写回所有折
#                          （load_ensemble 对各折阈值取均值，只有各折一致，
#                            集成实际生效的阈值才等于标定值）
#   2/5 全图评估       —— 有验证集：全折集成评官方验证集；
#                          没有：**留一折集成**（评估第 f 折时只用其余折的模型，
#                          避免"该折模型见过本折 val"造成的数据泄漏）
#   3/5 目标一/二 + 重复影像评估
#   4/5 规范 §5.2 导出提交权重
#   5/5 Mock Competition + 提交前检查清单
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FOLDS=(${FOLDS:-0 1 2})
EXIST=()
for f in "${FOLDS[@]}"; do
  [[ -f "checkpoints/g4_fold$f/best.pth" ]] && EXIST+=("$f") \
    || echo "[finalize] 跳过 fold$f（无 checkpoints/g4_fold$f/best.pth）"
done
[[ ${#EXIST[@]} -gt 0 ]] || { echo "[finalize] ✗ 没有任何可用折，先训练"; exit 2; }

_all() {   # 全部折的绝对路径（逗号分隔）
  local out=""
  for f in "${EXIST[@]}"; do out="${out:+$out,}$(realpath "checkpoints/g4_fold$f/best.pth")"; done
  echo "$out"
}
_except() {  # 排除第 $1 折的其余折（留一集成）
  local skip="$1" out=""
  for f in "${EXIST[@]}"; do
    [[ "$f" == "$skip" ]] && continue
    out="${out:+$out,}$(realpath "checkpoints/g4_fold$f/best.pth")"
  done
  [[ -n "$out" ]] && echo "$out" || realpath "checkpoints/g4_fold$skip/best.pth"
}

# ---- 官方验证集（可选链路）--------------------------------------------------
# 判定"用了哪套口径"只看**验证集清单**能否用（与评估脚本同一入口，不重复实现判定）：
#   有 → external（最终指标 = 官方验证集）；没有 → oof（折内 val）
# 说明：`assert_data_source` 的合规告警是 **stdout** 的普通 print，
# 会和探针结果混在一起，所以这里用 `VALMETA ` 前缀把机器可读行挑出来。
VAL_META="$("$PY" -c 'import sys
sys.path.insert(0, ".")
try:
    from src.utils.config import external_val_manifest
    man, path = external_val_manifest()
    n = sum(1 for c in man.get("cases") or [] if c.get("images")) if man else 0
    print("VALMETA %d %d %s" % (1 if man else 0, n, path))
except Exception:
    print("VALMETA 0 0 -")' 2>/dev/null | sed -n 's/^VALMETA //p' | tail -1 || true)"
HAS_VAL=0
if [[ "$VAL_META" == 1* ]]; then
  HAS_VAL=1
  echo "[finalize] 官方验证集：${VAL_META#1 }"
  echo "[finalize] → 采用 external 口径（全折集成，不做留一）"
else
  echo "[finalize] 未接入官方验证集（无 data/manifest_val.json）→ 回退折内 val（OOF/留一）"
  echo "[finalize]   想接入：export VAL_ROOT=<验证集目录> && bash scripts/01_probe.sh --val"
fi

# 自检用（由 scripts/24_verify_eval_split.py 调用）：
# 只打印"用哪些权重评哪些病例"的划分，不跑任何推理。
#   bash scripts/16_finalize.sh --print-split
# 输出（external）： split=external / external: g4_fold0,g4_fold1  ← 全部折
# 输出（oof）：      split=oof      / fold0: g4_fold1,g4_fold2     ← 排除 fold0 自身
if [[ "${1:-}" == "--print-split" ]]; then
  if [[ "$HAS_VAL" == "1" ]]; then
    echo "split=external"
    names=""
    for f in "${EXIST[@]}"; do names="${names:+$names,}g4_fold$f"; done
    echo "external: $names"
  else
    echo "split=oof"
    for f in "${EXIST[@]}"; do
      # 取权重所在**目录名**（各折文件名都叫 best.pth，取 basename 会全是 best.pth）
      names="$(tr ',' '\n' <<< "$(_except "$f")" | xargs -r -n1 dirname \
               | xargs -r -n1 basename | tr '\n' ',')"
      echo "fold$f: ${names%,}"
    done
  fi
  exit 0
fi

echo "==================== 1/5 集成阈值标定（写回所有折）===================="
if [[ "$HAS_VAL" == "1" ]]; then
  # 与最终评估同口径：官方验证集 + 全折集成（VAL_LIMIT 可限制标定例数）
  "$PY" scripts/14_calibrate_thresholds.py --split external --ckpt "$(_all)" \
    --limit "${VAL_LIMIT:-0}" --write
else
  "$PY" scripts/14_calibrate_thresholds.py --split fold --fold "${EXIST[0]}" --ckpt "$(_all)" \
    --limit 0 --write
fi

echo
if [[ "$HAS_VAL" == "1" ]]; then
  echo "==================== 2/5 全图评估（官方验证集，全折集成）===================="
  echo "--- 官方验证集：用全部 ${#EXIST[@]} 折集成（不做留一：验证集与训练集无交集）---"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$PY" -m src.evaluation.evaluate --split external --ckpt "$(_all)" 2>&1 \
    | tee "logs/eval_external_ens.log" | tail -20
else
  echo "==================== 2/5 全图评估（留一折集成，无泄漏）===================="
  for f in "${EXIST[@]}"; do
    ck="$(_except "$f")"
    n="$(awk -F, '{print NF}' <<< "$ck")"
    echo "--- fold$f：用其余 $n 折集成（排除 fold$f 自身）---"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      "$PY" -m src.evaluation.evaluate --split fold --fold "$f" --ckpt "$ck" 2>&1 \
      | tee "logs/eval_ens_fold$f.log" | tail -20
  done
fi

echo
if [[ "$HAS_VAL" == "1" ]]; then
  echo "==================== 3/5 目标一/二 + 重复影像评估（官方验证集，全折集成）===================="
  # 验证集自带 fake/拼接/重复对的金标准（在它自己的 special 里），
  # 且与训练集无交集 → 每个病例过全部折模型，无需 OOF。
  "$PY" scripts/15_eval_special_dup.py --split external --ckpt "$(_all)" 2>&1 | tail -30 || true
else
  echo "==================== 3/5 目标一/二 + 重复影像评估（OOF，无泄漏）===================="
  # 默认 OOF：逐折用本折模型只评本折 val，再汇总。
  # 本任务正样本极稀疏（624 例中 fake 6 / comp 6 / 重复对 12），
  # 单折 val 内只有 1~2 个正样本、重复对甚至为 0，指标会失去统计意义；
  # 全量又会让折模型评到自己的训练集（虚高）。OOF 同时解决这两点。
  "$PY" scripts/15_eval_special_dup.py --split oof 2>&1 | tail -30 || true
fi

echo
echo "==================== 4/5 按规范 §5.2 导出提交权重 ===================="
TAG_CSV="$(printf 'g4_fold%s,' "${EXIST[@]}")"; TAG_CSV="${TAG_CSV%,}"
bash scripts/09_export_submission.sh "$TAG_CSV"

if [[ "${SKIP_MOCK:-0}" != "1" ]]; then
  echo
  echo "==================== 5/5 Mock Competition（协议闭环）===================="
  bash scripts/08_mock_competition.sh
fi

echo
echo "==================== 提交前检查清单 ===================="
bash scripts/23_pre_submit_check.sh || true

cat <<'EOF'

------------------------------------------------------------------------
[finalize] 完成。提交要点：
  · 最终数字看哪份日志：external 口径 → logs/eval_external_ens.log（2/5 全折集成）
    + 3/5 的官方验证集输出；折内口径 → logs/eval_ens_fold*.log（留一折集成）。
    两者的阈值已在 1/5 用同一口径标定并写回所有折，不会互相打架。
  · 多折集成**无需手动指定权重**：resolve_checkpoints() 会自动读取
    /2026aicompetition/workspace/checkpoint/goal5_segmentation/ 下的全部 .pt。
  · 测评容器启动命令（启动脚本已内置 GLIOMA_LOADER_TOLERANT=1 容错）：
      CALLBACK_URL="<平台回调地址>" bash .../scripts/06_platform_serve.sh
  · 若检查清单有失败项，**不要提交**。
------------------------------------------------------------------------
EOF
