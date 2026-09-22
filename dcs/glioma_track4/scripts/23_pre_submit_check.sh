#!/usr/bin/env bash
# ============================================================================
# 提交前检查清单（一条命令跑完全部验证）
#
#   bash scripts/23_pre_submit_check.sh            # 全量（含 GPU 项，约 5~10 分钟）
#   bash scripts/23_pre_submit_check.sh --quick    # 快速（跳过耗时项）
#
# 检查项与"为什么需要它"：
#   ① 语法/导入        —— 提交物里任何语法或 import 错误都会让容器起不来
#   ② 团队按钮测试      —— 确认我改过的公共代码没有破坏团队的既有契约
#   ③ P0 修复断言       —— 7 个致命缺陷（TTA 双重翻折/嵌入损失未生效/掩码 header 污染…）
#   ④ 端到端自检        —— 训练+推理+答案格式全链路
#   ⑤ 桥接集成          —— 我的插件与团队链路的真实对接
#   ⑥ 故障注入压测      —— 22 类异常输入，任何一例都必须仍有合规答案
#   ⑦ Mock Competition —— 平台协议闭环（/health → /call → 后台 → 回调 → 校验）
#   ⑧ 提交边界          —— .gitignore 不能让源码（尤其 src/data/）被忽略
#   ⑨ 权重导出复核      —— 规范 §5.2 路径下权重齐备
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
TEAM="${TEAM_ROOT:-../Glioma_recognition-main}"
PY="${PYTHON:-python}"
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

PASS=0; FAIL=0; SKIP=0
declare -a REPORT=()

_ok()   { PASS=$((PASS+1)); REPORT+=("  ✓ $1"); printf '  ✓ %s\n' "$1"; }
_bad()  { FAIL=$((FAIL+1)); REPORT+=("  ✗ $1"); printf '  ✗ %s\n' "$1"; }
_skip() { SKIP=$((SKIP+1)); REPORT+=("  - $1（跳过）"); printf '  - %s（跳过）\n' "$1"; }

_sec() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# --------------------------------------------------------------------------- #
_sec "① 语法编译 + 关键模块导入"
if find src integration scripts -name '*.py' -not -path '*__pycache__*' -print0 2>/dev/null \
     | xargs -0 -r "$PY" -m py_compile 2>/dev/null; then
  _ok "全部 .py 语法通过"
else
  _bad "存在语法错误"
fi
if "$PY" - <<'PYEOF' >/dev/null 2>&1
import sys
sys.path.insert(0, "."); sys.path.insert(0, "../Glioma_recognition-main")
import importlib
for m in ("src.inference.pipeline", "src.inference.writer", "src.data.dataset",
          "src.data.dicom", "integration.tasks", "integration.factory"):
    importlib.import_module(m)
PYEOF
then _ok "关键模块导入通过"; else _bad "模块导入失败"; fi

# --------------------------------------------------------------------------- #
_sec "② 团队按钮测试（既有契约）"
if [[ -d "$TEAM" ]]; then
  if (cd "$TEAM" && "$PY" -m pytest tests/ -q 2>&1 | tail -1 | grep -q passed); then
    _ok "团队测试通过（含 fail-fast 契约断言）"
  else
    _bad "团队测试失败"
  fi
else
  _bad "找不到团队仓库 $TEAM"
fi

# --------------------------------------------------------------------------- #
_sec "③ P0 修复断言"
if out=$("$PY" scripts/21_verify_fixes.py 2>&1 | tail -3); then
  if grep -q "全部 PASS" <<< "$out"; then _ok "P0 修复 16/16 断言通过"; else _bad "P0 断言未全通过"; fi
else
  _bad "P0 断言执行失败"
fi

# --------------------------------------------------------------------------- #
if [[ $QUICK -eq 1 ]]; then
  _sec "④~⑦ 端到端 / 桥接 / 故障注入 / Mock（--quick 跳过）"
  _skip "端到端自检"; _skip "桥接集成"; _skip "故障注入压测"; _skip "Mock Competition"
else
  _sec "④ 端到端自检"
  if "$PY" scripts/99_smoke_test.py 2>&1 | tail -5 | grep -q "PASS"; then
    _ok "端到端自检 PASS"
  else
    _bad "端到端自检失败"
  fi

  _sec "⑤ 桥接集成"
  if "$PY" scripts/20_bridge_selftest.py --n 3 2>&1 | grep -q "\[bridge\] PASS"; then
    _ok "桥接集成 PASS"
  else
    _bad "桥接集成失败"
  fi

  _sec "⑥ 故障注入压测（22 类异常输入）"
  if "$PY" scripts/22_fault_injection.py 2>&1 | tail -8 | grep -q "全部通过"; then
    _ok "故障注入 22/22 均有合规答案"
  else
    _bad "故障注入存在未产出答案的病例"
  fi

  _sec "⑦ Mock Competition（平台协议闭环）"
  if bash scripts/08_mock_competition.sh 2>&1 | grep -q "PASS Callback"; then
    _ok "Mock Competition 协议闭环通过"
  else
    _bad "Mock Competition 失败"
  fi
fi

# --------------------------------------------------------------------------- #
_sec "⑧ 提交边界（源码不得被 .gitignore 吞掉）"
if [[ -f .gitignore ]]; then
  _ok ".gitignore 存在"
  tmp="$(mktemp -d)"
  mkdir -p "$tmp/src/data" "$tmp/data" "$tmp/checkpoints" "$tmp/logs"
  cp .gitignore "$tmp/"
  touch "$tmp/src/data/dataset.py" "$tmp/data/manifest.json" \
        "$tmp/checkpoints/best.pth" "$tmp/logs/train.log"
  if (cd "$tmp" && git init -q 2>/dev/null && git add -A 2>/dev/null \
        && git ls-files --error-unmatch src/data/dataset.py >/dev/null 2>&1); then
    _ok "源码 src/data/ 正常入库（未被忽略）"
  else
    _bad "源码 src/data/ 被 .gitignore 忽略 —— 提交后必崩！"
  fi
  if (cd "$tmp" && git ls-files --error-unmatch data/manifest.json >/dev/null 2>&1); then
    _bad "本地产物 data/ 被误入库"
  else
    _ok "本地产物 data/ 已正确排除"
  fi
  rm -rf "$tmp"
else
  _bad "缺少 .gitignore（权重/产物会被一起提交）"
fi

# 硬编码本地路径：**只查运行时代码**（src/ + integration/）。
# scripts/ 下的自测脚本允许有本地默认值（均可通过 CLI 参数覆盖），且不参与提交运行，
# 因此仅提示、不计失败；同时排除本脚本自身（它含有待检测的字面量）。
hits="$(grep -rn "/mnt/\|/home/[a-z]\{1,\}/" --include=*.py src integration 2>/dev/null \
        | grep -v "__pycache__" || true)"
if [[ -n "$hits" ]]; then
  _bad "运行时代码存在硬编码本地路径（容器内不存在）"
  echo "$hits" | head -3
else
  _ok "运行时代码无硬编码本地路径"
fi
soft="$(grep -rln "/mnt/\|/home/[a-z]\{1,\}/" --include=*.py --include=*.sh scripts 2>/dev/null \
        | grep -v "23_pre_submit_check" || true)"
if [[ -n "$soft" ]]; then
  echo "  · 提示：自测脚本含本地默认路径（可 CLI 覆盖，不影响提交）："
  echo "$soft" | sed 's/^/      /'
fi

# --------------------------------------------------------------------------- #
_sec "⑨ 评估划分逻辑自检（OOF / 留一折）"
if out=$("$PY" scripts/24_verify_eval_split.py 2>&1); then
  if grep -q "全部 PASS" <<< "$out"; then
    _ok "评估划分逻辑自检通过（含 OOF 语义端到端验证）"
  else
    _bad "评估划分自检未全通过"
  fi
  if grep -q "WARN" <<< "$out"; then
    echo "  · 提示（已知偏差，评估侧已处理）："
    grep "WARN" <<< "$out" | sed 's/^/      /'
  fi
else
  _bad "评估划分自检失败"
  grep -E "FAIL" <<< "$out" | head -3
fi

# --------------------------------------------------------------------------- #
_sec "⑩ 权重导出复核（规范 §5.2）"
WS="${WORKSPACE:-/2026aicompetition/workspace}"
export GLIOMA_CHECKPOINT_ROOT="${GLIOMA_CHECKPOINT_ROOT:-$WS/checkpoint}"
# 注意：不能用 `export_ckpt --verify` 判定——它走的是 resolve_checkpoints()，
# 含"本地 checkpoints/g4_fold*"回退，权重**没导出也会返回 0**。
# 这里直接检查**规范路径下是否真有 .pt 文件**。
n_w="$(find "$GLIOMA_CHECKPOINT_ROOT/goal5_segmentation" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)"
if [[ "$n_w" -gt 0 ]]; then
  _ok "权重已导出（goal5_segmentation 下 $n_w 个 .pt）"
  find "$GLIOMA_CHECKPOINT_ROOT/goal5_segmentation" -maxdepth 1 -name '*.pt' 2>/dev/null | sed 's/^/      /'
else
  _bad "权重未导出：$GLIOMA_CHECKPOINT_ROOT/goal5_segmentation/ 下没有 .pt（提交前必须跑 scripts/09_export_submission.sh）"
fi
for g in goal1_authenticity goal2_stitched goal2_duplicate goal3_tumor goal4_diagnosis; do
  n="$(find "$GLIOMA_CHECKPOINT_ROOT/$g" -maxdepth 1 -type f 2>/dev/null | wc -l)"
  [[ "$n" -gt 0 ]] && echo "      · $g: $n 个文件" || echo "      ! $g: 空（未导出）"
done

# --------------------------------------------------------------------------- #
printf '\n%0.s=' {1..66}; echo
printf '汇总：%d 通过 / %d 失败 / %d 跳过\n' "$PASS" "$FAIL" "$SKIP"
printf '%0.s=' {1..66}; echo
for line in "${REPORT[@]}"; do echo "$line"; done
echo
if [[ $FAIL -eq 0 ]]; then
  echo "✔ 提交就绪"
  exit 0
else
  echo "✘ 存在 $FAIL 项失败 —— 不要提交"
  exit 1
fi
