#!/usr/bin/env python
"""桥接自测：用**团队串联工程**的 Loader/Runner/Writer/Validator 跑通本工程插件。

这是"能否衔接提交"的最硬验证——不使用本工程任何自研的服务/写出/校验代码，
只把算法封装成团队契约的 ``StudyTask``/``DatasetTask``，由团队链路端到端执行：

    DatasetLoader.iter_studies → InferencePipeline.run_study（本工程插件）
      → OutputWriter.write_study → OutputValidator.validate_study → publish

用法：
    python scripts/20_bridge_selftest.py [--team <Glioma_recognition 路径>] [--n 3]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG_NAMES = ("t1c.nii.gz", "flair.nii.gz", "t2.nii.gz", "t1.nii.gz")


def build_eval_dataset(src_root: Path, dst_root: Path, n: int) -> list[str]:
    """构造**只含影像**的测试集（团队 Loader 会把含 mask/seg/label/roi 的文件排除，
    但本工程的合成数据掩码是中文名，必须显式只复制影像）。"""
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
    accs = []
    for entry in sorted(src_root.iterdir()):
        if len(accs) >= n:
            break
        if not entry.is_dir() or entry.name == "annotation":
            continue
        copied = False
        for series_dir in sorted(entry.iterdir()):
            if not series_dir.is_dir():
                continue
            target = dst_root / entry.name / series_dir.name
            for name in IMG_NAMES:
                source = series_dir / name
                if source.is_file():
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target / name)
                    copied = True
        # 团队 Loader 要求每个 series 目录内**只有一个** NIfTI 且与目录同名，
        # 因此把 <mod>.nii.gz 重命名为 <series_uid>.nii.gz
        if copied:
            for series_dir in (dst_root / entry.name).iterdir():
                files = list(series_dir.glob("*.nii.gz"))
                if len(files) == 1 and files[0].stem != series_dir.name:
                    files[0].rename(series_dir / f"{series_dir.name}.nii.gz")
            accs.append(entry.name)
    return accs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=str(ROOT.parent / "Glioma_recognition-main"))
    ap.add_argument("--src", default=os.environ.get("DATASET_ROOT")
                    or "/mnt/data_sdb/wangx/data/Brain_MRI/track4_sim")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ckpt", default=None, help="默认取 checkpoints/g4_fold*/best.pth 首折")
    ap.add_argument("--work", default=None)
    ap.add_argument("--goals", default="goal1,goal2_stitched,goal3,goal5,goal4,goal2_duplicate",
                    help="启用的真实插件（其余用团队 Dummy 补位）")
    a = ap.parse_args()

    team = Path(a.team).resolve()
    if not (team / "core" / "runner.py").is_file():
        print(f"[bridge] ✗ 未找到团队工程：{team}")
        return 1
    sys.path.insert(0, str(team))
    os.environ.setdefault("GLIOMA_TRACK4_ROOT", str(ROOT))

    ckpt = a.ckpt
    if not ckpt:
        found = sorted((ROOT / "checkpoints").glob("g4_fold*/best.pth"))
        found += sorted((ROOT / "checkpoints").glob("g4L_fold*/best.pth"))
        if not found:
            print("[bridge] ✗ 未找到权重：先训练（bash scripts/03_train.sh 0）")
            return 1
        ckpt = str(found[0])
    os.environ["GLIOMA_CKPT"] = ckpt
    os.environ["GLIOMA_GOALS"] = a.goals
    print(f"[bridge] 算法权重: {ckpt}")
    print(f"[bridge] 启用插件: {a.goals}")

    work = Path(a.work or tempfile.mkdtemp(prefix="glioma_bridge_"))
    dataset = work / "dataset"
    accs = build_eval_dataset(Path(a.src), dataset, a.n)
    print(f"[bridge] 测试集 {len(accs)} 例: {accs}  ->  {dataset}")

    from core.config import Settings
    from core.runner import EvaluationJob, EvaluationRunner

    answer_root = work / "answer"
    # 关键：通过团队约定的工厂入口装配真实插件（等价于平台上的
    # COMPETITION_PIPELINE_FACTORY=tasks.glioma.pipeline:build_pipeline）
    settings = Settings(workspace=work, answer_root=answer_root, log_root=work / "logs",
                        callback_url=None,
                        pipeline_factory="tasks.glioma.pipeline:build_pipeline")
    runner = EvaluationRunner(settings)
    pipeline = runner.pipeline
    print("[bridge] 插件绑定: " + ", ".join(
        f"{b.context_field}={type(b.task).__name__}" for b in pipeline.study_tasks
    ) + f"; duplicate={type(pipeline.duplicate_task).__name__}")

    out = runner.run(
        EvaluationJob(request_id="bridge-selftest", evaluation_id="bridge_0001",
                      dataset_path=dataset),
        send_callback=False,
    )
    print(f"[bridge] 发布目录: {out}")

    # ---- 独立复核：目录结构 / JSON / 掩码 / duplicate ----
    import json

    import nibabel as nib
    import numpy as np

    errors: list[str] = []
    for acc in accs:
        pj = out / acc / "prediction.json"
        if not pj.is_file():
            errors.append(f"{acc}: 缺 prediction.json")
            continue
        payload = json.loads(pj.read_text(encoding="utf-8"))
        uri = payload.get("SegmentationMaskURI") or {}
        if set(uri) != {"core", "flair"}:
            errors.append(f"{acc}: SegmentationMaskURI 键不是 core/flair")
        for key, rel in uri.items():
            mp = (out / acc / str(rel)).resolve()
            if not mp.is_file():
                errors.append(f"{acc}:{key} 掩码不存在 {rel}")
                continue
            arr = np.asanyarray(nib.load(str(mp)).dataobj)
            src = nib.load(str(dataset / acc / mp.parent.name / f"{mp.parent.name}.nii.gz"))
            if arr.shape != src.shape:
                errors.append(f"{acc}:{key} shape 不一致")
            if not np.allclose(nib.load(str(mp)).affine, src.affine, atol=1e-5):
                errors.append(f"{acc}:{key} affine 超容差(1e-5)")
            if not set(np.unique(arr).tolist()) <= {0, 1}:
                errors.append(f"{acc}:{key} 非二值")

    dp = out / "duplicate_pairs.jsonl"
    n_pairs = 0
    if dp.is_file():
        for line in dp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if set(record) != {"StudyUID", "StudyUID_dup", "PairProb"}:
                    errors.append("duplicate 字段不对")
                if record["StudyUID"] == record["StudyUID_dup"]:
                    errors.append("duplicate 出现 self-pair")
                n_pairs += 1
    else:
        errors.append("缺 duplicate_pairs.jsonl")

    print("=" * 70)
    if errors:
        print("[bridge] FAIL ✗")
        for e in errors[:10]:
            print("   -", e)
    else:
        print(f"[bridge] PASS ✔ 团队链路 + 本工程插件端到端通过"
              f"（{len(accs)} 例，掩码/JSON/duplicate 全部合规，duplicate {n_pairs} 行）")
    print(f"[bridge] 工作目录保留在 {work}（可自行清理）")
    print("=" * 70)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
