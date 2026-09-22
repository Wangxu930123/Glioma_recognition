#!/usr/bin/env bash
# =============================================================================
# 训练工程 → 提交工程 的**权重交付**（规范 §5.2）
# =============================================================================
# 两个工程是**不合并**的，本脚本是它们之间唯一的交付动作：
#
#   glioma_goals/<goal>/runs/<tag>/checkpoints/best.pth
#        │  ← 本脚本（复制 + 校验元信息 + 改名到规范文件名）
#        ▼
#   <workspace>/checkpoint/<goal>/model.pt
#        │  ← 提交工程在**服务启动时**读取（tasks/<goal>/config.py 里的 ckpt_rel）
#        ▼
#   Docker 镜像内的推理进程
#
# 为什么不把权重放进代码仓库：规范 §5.2 明确"权重不放在代码仓库，
# 统一位于 /2026aicompetition/workspace/checkpoint/"；且 6 份权重
# 合计数十 MB，打进镜像会让每次迭代都重新传一遍。
#
# 用法::
#
#   bash scripts/export_to_submission.sh                      # 各 Goal 用同名 tag
#   TAG=exp1 bash scripts/export_to_submission.sh             # 所有 Goal 用同一 tag
#   WORKSPACE=/tmp/ws bash scripts/export_to_submission.sh    # 演练（不碰真实路径）
#   GOALS="goal5_segmentation" TAG=exp1 bash scripts/...      # 只导一个
#   VERIFY=0 bash scripts/...                                 # 跳过元信息校验
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${WORKSPACE:-/2026aicompetition/workspace}"
DEST_ROOT="${GLIOMA_CHECKPOINT_ROOT:-$WS/checkpoint}"
TAG="${TAG:-}"
GOALS="${GOALS:-goal1_authenticity goal2_stitched goal2_duplicate goal3_tumor goal4_diagnosis goal5_segmentation}"
PY="${PYTHON:-python3}"
VERIFY="${VERIFY:-1}"

echo "源工程    : $ROOT"
echo "目标根目录: $DEST_ROOT"
echo "tag       : ${TAG:-<各 Goal 同名目录>}"
echo

ok=0; miss=0
for g in $GOALS; do
  src="$ROOT/$g/runs/${TAG:-$g}/checkpoints/best.pth"
  dst_dir="$DEST_ROOT/$g"

  if [[ ! -f "$src" ]]; then
    printf "  %-20s ✗ 缺权重：%s\n" "$g" "$src"
    printf "  %-20s   （先训练：cd %s && python train.py --tag %s）\n" "" "$g" "${TAG:-$g}"
    miss=$((miss + 1))
    continue
  fi

  mkdir -p "$dst_dir"

  # 规范 §5.2 的**文件名**约定：goal2_duplicate 叫 encoder.pt，其余叫 model.pt；
  # goal5 需要 core.pt（+ flair.pt，见下方说明）。
  case "$g" in
    goal2_duplicate) names="encoder.pt" ;;
    goal5_segmentation) names="core.pt flair.pt" ;;
    *) names="model.pt" ;;
  esac

  for n in $names; do
    cp -f "$src" "$dst_dir/$n"
  done

  # 权重元信息必须齐全——缺了它推理侧只能按默认值重建网络，
  # 形状不符的层会被 strict=False **静默跳过**（最难查的一种失效）。
  if [[ "$VERIFY" == "1" ]]; then
    info="$("$PY" - "$dst_dir/${names%% *}" <<'PYEOF'
import sys
import torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
need = ("model_ema", "cls_spec", "arch", "model_cfg")
lack = [k for k in need if k not in ck]
if lack:
    raise SystemExit(f"缺元信息 {lack}")
heads = len([k for k in ck["model_ema"] if k.startswith("cls_heads.")])
print(f"{heads // 2} 个分类头 / arch={ck['arch']} / "
      f"in_ch={(ck.get('model_cfg') or {}).get('in_ch')}")
PYEOF
)" || info=""
    if [[ -z "$info" ]]; then
      printf "  %-20s ✗ 权重元信息不完整（推理侧会静默加载错结构）\n" "$g"
      miss=$((miss + 1))
      continue
    fi
    printf "  %-20s ✓ → %s/%s（%s）\n" "$g" "$g" "$names" "$info"
  else
    printf "  %-20s ✓ → %s/%s\n" "$g" "$g" "$names"
  fi
  ok=$((ok + 1))
done

echo
echo "成功 $ok 个 / 缺失 $miss 个"
if [[ "$miss" -gt 0 ]]; then
  echo
  echo "提示：goal4_diagnosis 需要数据带 label.json（14 个结构化字段）才能训练；"
  echo "      其余 Goal 无标注也能跑，但目标一/二/三的正样本极稀疏。"
  exit 1
fi

cat <<'NOTE'

下一步（三选一）：
  ① 带真实权重的完整演练（推荐先跑这个）
       cd ../Glioma_recognition-main
       COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline \
       python scripts/mock_competition.py --workspace "$DEST_ROOT/.."

  ② 提交前检查清单
       cd ../glioma_track4 && bash scripts/23_pre_submit_check.sh

  ③ 打镜像提交
       权重放在平台私有存储，**不要**打进镜像；镜像里只含 Glioma_recognition-main/

说明：goal5 的 flair.pt 当前与 core.pt 同源——提交侧的 load_model() 只读
      core_ckpt_rel（flair_ckpt_rel 是为"core/flair 分开训练"预留的配置项）。
      两个文件都写出，是为了与规范 §5.2 的目录约定保持一致。
NOTE
