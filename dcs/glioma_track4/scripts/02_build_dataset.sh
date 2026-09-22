#!/usr/bin/env bash
# 构建训练集清单与多折划分（探针已在 01 步产出 manifest；此处只做折划分与统计）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-python}"
N_FOLDS="${N_FOLDS:-5}"   # 注意：不要用 FOLDS，它与 paths.yaml 的环境变量覆盖同名

"$PY" - <<PY
import json
from src.utils.config import load_paths, resolve
from src.data.dataset import build_folds
paths = load_paths()
man = resolve(paths["manifest"])
import os
assert os.path.exists(man), f"缺 manifest：{man}（先跑 bash scripts/01_probe.sh）"
d = json.load(open(man, encoding="utf-8"))
n = len(d["cases"])
pos = sum(1 for c in d["cases"] if c.get("masks") or c.get("labels"))
print(f"[02] 病例 {n} 例（含标注/掩码 {pos}）")
folds = build_folds(paths["manifest"], n_folds=$N_FOLDS, val_ratio=0.2)
for k, v in folds.items():
    print(f"  fold{k}: train {len(v['train'])} / val {len(v['val'])}")
print(f"[02] 折文件 -> {resolve(paths['folds'])}")
PY
echo "[02] 完成。下一步：bash scripts/03_train.sh 0"
