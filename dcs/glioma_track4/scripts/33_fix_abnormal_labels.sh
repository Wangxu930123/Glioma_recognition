#!/usr/bin/env bash
# 33_fix_abnormal_labels.sh —— 一条命令修好 1_abnormal 的 Label（来源）列，并重跑探针
#
# 做四件事（每步都会打印它干了什么）：
#   ① 定位：找到 labels/1_abnormal.xlsx（原表）、人工补丁（如 1_abnormal_wzh.xlsx）、
#      以及**病例层**数据根（自动下钻到 .../annotation）
#   ② 核验：按 Label 拼路径做磁盘核对，报"改前 → 改后"命中率（这一步不写文件）
#   ③ 合并：命中率上升才落地，自动备份为 1_abnormal.bak-<时间>.xlsx
#   ④ 重跑：export GLIOMA_LABELS_DIR 后跑 scripts/01_probe.sh
#
# 用法：
#   bash scripts/33_fix_abnormal_labels.sh                  # 全自动：定位→核验→落地→探针
#   bash scripts/33_fix_abnormal_labels.sh --dry-run        # 只定位+核验，不写文件、不跑探针
#   bash scripts/33_fix_abnormal_labels.sh --no-probe       # 合并但不重跑探针
#   bash scripts/33_fix_abnormal_labels.sh --labels <目录> --patch <文件> --root <数据根>
#
# 退出码：0 成功；2 定位/参数有问题；3 补丁命中率更低（拒绝写回）；4 --root 指错层
#        （4 时把 --root 指到"直接含检查号目录与 fake/compositing/duplicate 的那层"重跑）
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LABELS_ARG=""; PATCH_ARG=""; ROOT_ARG=""
DO_APPLY=1; RUN_PROBE=1; FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --labels)  LABELS_ARG="${2:-}"; shift 2 ;;
        --patch)   PATCH_ARG="${2:-}"; shift 2 ;;
        --root)    ROOT_ARG="${2:-}"; shift 2 ;;
        --dry-run) DO_APPLY=0; RUN_PROBE=0; shift ;;
        --no-probe) RUN_PROBE=0; shift ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[33] 未知参数：$1（用 --help 看用法）" >&2; exit 2 ;;
    esac
done

# ---- python 解释器（容器里 python3 / python 都可能缺席一个）----
PY=""
for c in "${PYTHON:-}" python3 python; do
    [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "[33] 找不到 python 解释器" >&2; exit 2; }

OFFICIAL_5="1_abnormal 2_duplicate 3_serieslabel 4_masklabel 5_characteristics"
WORKSPACE="${WORKSPACE:-/2026aicompetition/workspace}"

# --------------------------------------------------------------------------- #
# ① 定位 labels 目录（原表所在）
#    优先级：显式参数 > $GLIOMA_LABELS_DIR > 仓库内 labels/ > 工作区搜索
#    同名多份时：官方 5 张表更全的优先；data_validation/ 下的副本降权
# --------------------------------------------------------------------------- #
find_labels_dir() {
    local cands=() d p
    for d in "$LABELS_ARG" "${GLIOMA_LABELS_DIR:-}" "$REPO_ROOT/labels" "$REPO_ROOT/../labels"; do
        [ -n "$d" ] && [ -f "$d/1_abnormal.xlsx" ] && cands+=("$d")
    done
    if [ ${#cands[@]} -eq 0 ] && [ -d "$WORKSPACE" ]; then
        while IFS= read -r p; do
            [ -n "$p" ] && cands+=("$(dirname "$p")")
        done < <(find "$WORKSPACE" -maxdepth 6 -name "1_abnormal.xlsx" 2>/dev/null | sort)
    fi
    local best="" best_score=-1 best_mt=0 n score mt f
    for d in "${cands[@]}"; do
        n=0
        for f in $OFFICIAL_5; do [ -f "$d/$f.xlsx" ] && n=$((n + 1)); done
        score=$((n * 100))
        case "$d" in */data_validation/*) score=$((score - 50)) ;; esac   # 降权，非首选
        mt=$(stat -c %Y "$d/1_abnormal.xlsx" 2>/dev/null || echo 0)
        if [ "$score" -gt "$best_score" ] || { [ "$score" -eq "$best_score" ] && [ "$mt" -gt "$best_mt" ]; }; then
            best="$d"; best_score="$score"; best_mt="$mt"
        fi
    done
    printf '%s\n' "$best"
}

# ② 找人工补丁（名字里带 wzh 或 1_abnormal_ 前缀，取最新）
find_patch() {
    local dir="$1" cands=() p
    for p in "$PATCH_ARG" "${ABNORMAL_PATCH:-}"; do
        [ -n "$p" ] && [ -f "$p" ] && cands+=("$p")
    done
    if [ ${#cands[@]} -eq 0 ]; then
        while IFS= read -r p; do [ -n "$p" ] && cands+=("$p"); done < <(
            ls -1 "$dir"/../*wzh*.xlsx "$dir"/*wzh*.xlsx "$dir"/../1_abnormal_*.xlsx \
                  2>/dev/null | sort -u
            find "$WORKSPACE" -maxdepth 6 \( -name "*wzh*.xlsx" -o -name "1_abnormal_*.xlsx" \) \
                2>/dev/null | sort -u
        )
    fi
    local best="" best_mt=0 mt
    for p in "${cands[@]}"; do
        [ "$p" = "$dir/1_abnormal.xlsx" ] && continue
        mt=$(stat -c %Y "$p" 2>/dev/null || echo 0)
        [ "$mt" -gt "$best_mt" ] && { best="$p"; best_mt="$mt"; }
    done
    printf '%s\n' "$best"
}

# ③ 病例层：数据根 → 若含 annotation/ 则下钻（32 里还会再兜一层）
find_annot_root() {
    local r="${ROOT_ARG:-${DATASET_ROOT:-/2026aicompetition/datasets/training}}"
    if [ -d "$r/annotation" ]; then printf '%s\n' "$r/annotation"; else printf '%s\n' "$r"; fi
}

LABELS_DIR="$(find_labels_dir)"
[ -n "$LABELS_DIR" ] || {
    echo "[33] 找不到 labels/1_abnormal.xlsx" >&2
    echo "     试：bash scripts/33_fix_abnormal_labels.sh --labels <那个 labels 目录>" >&2
    echo "     或在容器里：find /2026aicompetition -name '1_abnormal.xlsx' 2>/dev/null" >&2
    exit 2
}
PATCH_FILE="$(find_patch "$LABELS_DIR")"
[ -n "$PATCH_FILE" ] || {
    echo "[33] 找不到人工补丁（*wzh*.xlsx / 1_abnormal_*.xlsx）" >&2
    echo "     试：bash scripts/33_fix_abnormal_labels.sh --patch <补丁文件>" >&2
    exit 2
}
ANNOT_ROOT="$(find_annot_root)"

echo "=========================================================================="
echo "[33] 定位结果"
echo "     原表 base  = $LABELS_DIR/1_abnormal.xlsx"
echo "     补丁 patch = $PATCH_FILE"
echo "     病例层根   = $ANNOT_ROOT"
off=0; for f in $OFFICIAL_5; do [ -f "$LABELS_DIR/$f.xlsx" ] && off=$((off + 1)); done
echo "     labels 目录里官方 5 张表：$off/5"
case "$LABELS_DIR" in
    */data_validation/*) echo "     ⚠ 用的是 data_validation/ 下的暂存副本，建议改用团队工作区那份" ;;
esac
echo "=========================================================================="
echo

# --------------------------------------------------------------------------- #
# ② 核验（dry-run，不写任何文件）
# --------------------------------------------------------------------------- #
STEP2=("$PY" "$REPO_ROOT/scripts/32_apply_abnormal_patch.py"
       --base "$LABELS_DIR/1_abnormal.xlsx" --patch "$PATCH_FILE" --root "$ANNOT_ROOT")
"${STEP2[@]}"
rc=$?
if [ $rc -eq 4 ]; then
    echo "[33] 中止：--root 指错了层。请指到**直接含检查号目录与 fake/compositing/duplicate 的那层**" >&2
    echo "     例：bash scripts/33_fix_abnormal_labels.sh --root /2026aicompetition/datasets/training/annotation" >&2
    exit 4
fi
[ $rc -eq 0 ] || { echo "[33] 中止：核验失败（退出码 $rc）" >&2; exit $rc; }

if [ "$DO_APPLY" -eq 0 ]; then
    echo
    echo "[33] --dry-run：只核验，未写文件、未跑探针。去掉 --dry-run 即可落地。"
    exit 0
fi

# --------------------------------------------------------------------------- #
# ③ 落地（命中率把关 + 自动备份都在 32 里）
# --------------------------------------------------------------------------- #
echo
echo "[33] 落地合并 …"
APPLY=("${STEP2[@]}" --apply)
[ "$FORCE" -eq 1 ] && APPLY+=(--force)
"${APPLY[@]}"
rc=$?
if [ $rc -eq 3 ]; then
    echo "[33] 中止：补丁命中率比原表低 → 拒绝写回。若你确认补丁才是对的，加 --force 重跑" >&2
    exit 3
fi
[ $rc -eq 0 ] || { echo "[33] 中止：合并失败（退出码 $rc）" >&2; exit $rc; }

# --------------------------------------------------------------------------- #
# ④ 重跑探针（表目录已在工作区，必须显式告诉探针）
# --------------------------------------------------------------------------- #
export GLIOMA_LABELS_DIR="$LABELS_DIR"
echo
echo "[33] export GLIOMA_LABELS_DIR=$GLIOMA_LABELS_DIR"
if [ "$RUN_PROBE" -eq 1 ]; then
    echo "[33] 重跑探针 …"
    echo "--------------------------------------------------------------------------"
    ( cd "$REPO_ROOT" && bash scripts/01_probe.sh )
    echo "--------------------------------------------------------------------------"
    echo "[33] 看探针输出里的三处："
    echo "     · official_label_files  → 应列出 5 张表（不再是 {{}}）"
    echo "     · abnormal_label_counts → 应出现 compositing / duplicate 的计数"
    echo "     · label_field_counts    → 结构化字段应有值（金标准改取 5_characteristics.xlsx）"
else
    echo "[33] --no-probe：已跳过。手动重跑："
    echo "     export GLIOMA_LABELS_DIR=$GLIOMA_LABELS_DIR && bash scripts/01_probe.sh"
fi
echo "[33] 完成。"
