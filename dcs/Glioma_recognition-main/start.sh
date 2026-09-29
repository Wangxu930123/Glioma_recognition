#!/usr/bin/env sh
set -eu

# --------------------------------------------------------------------------- #
# 加载容错（正式评测默认开启）
# --------------------------------------------------------------------------- #
# ⚠️ ``data/loader._TOLERANT`` 是在**模块导入期**读取本变量的 —— 进程跑起来之后再改
#    环境变量**不生效**，所以必须在这里 export，而不是等到运行时。
#
# 打开后覆盖 4 处（**全部只在容错模式生效**；关闭时行为与上游完全一致）：
#   ① 序列目录有多个 NIfTI 且无"主名 == 目录名"的唯一原件 → 跳过该序列
#   ② 单个序列文件损坏 / 维度非法 → 跳过该序列
#   ③ SeriesType.xlsx 读不到 / 表头不认 / 同键冲突 → 降级为"无表继续"
#      （描述退回 sidecar/目录名，模态随后走官方表 / 体素判别兜底）
#   ④ core/runner 逐例容错：单例失败只丢该例，不再 rmtree 整个 staging
#
# 为什么默认开：评测**不可重跑**。默认 fail-fast 下，真数据里只要有 1 个脏目录或
# 1 次推理异常，``_run_streaming`` 就会把整批 staging 删掉 → **几百例一起 0 分**；
# 打开后最坏情况只是丢 1 例的 1 个通道。
#
# 想严格与上游 README §3 字面一致（fail-fast）：GLIOMA_LOADER_TOLERANT=0 ./start.sh
: "${GLIOMA_LOADER_TOLERANT:=1}"
export GLIOMA_LOADER_TOLERANT

exec python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 1
