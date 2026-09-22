#!/usr/bin/env bash
# 数据探针：实测比赛训练集结构 → data/manifest.json + 结构报告
# 用法：bash scripts/01_probe.sh [数据根]        （数据根可省略，默认取 configs/paths.yaml）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-python}"
ARGS=()
[[ $# -ge 1 ]] && ARGS+=(--root "$1")
"$PY" -m src.data.probe "${ARGS[@]}"
echo
echo "[01] 请检查报告中的这些点（决定后续映射是否要改）："
echo "  · modality_counts：是否识别到 t1c / flair / t2（缺哪个就看 fallback 是否够用）"
echo "  · mask_role_counts：core / peri 是否都识别到（regions 命名是否命中关键词）"
echo "  · label_field_counts：结构化字段命中数（0 = 没找到金标准表，需指定）"
echo "  · missing_t1c / missing_flair：缺序列的病例数（训练/推理会自动降级）"
echo "  · special.gold_pairs：重复影像金标准对数（0 = 格式不匹配，需按实际调整）"
