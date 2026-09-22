"""Goal5 插件契约测试（规范 §19.2 / §18.3）。

只验证**接口契约与约束**，不依赖真实训练权重，因此可在 CI 中稳定运行：
1. 能实例化并加载模型；
2. 权重只能从 ``{ckpt_root}/<relative>`` 解析，不允许路径逃逸；
3. 权重与配置结构不兼容时必须**明确失败**，而不是静默加载随机层；
4. 返回约定的强类型 ``Goal5Result``；
5. 不产生任何比赛目录副作用（不写 answer/、不发回调）；
6. 比赛入口不传递导入训练专用模块。
"""
from __future__ import annotations

import ast
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Settings                                    # noqa: E402
from data.structures import Series, Study                           # noqa: E402
from tasks.goal5_segmentation.config import Goal5Config             # noqa: E402
from tasks.goal5_segmentation.inference import resolve_ckpt         # noqa: E402
from tasks.results import Goal5Result                               # noqa: E402


# --------------------------------------------------------------------------- #
# 1. 路径解析：只能落在 checkpoint 根目录内
# --------------------------------------------------------------------------- #
def test_resolve_ckpt_stays_inside_root(tmp_path):
    got = resolve_ckpt(tmp_path, "goal5_segmentation/core.pt")
    assert got == (tmp_path / "goal5_segmentation" / "core.pt").resolve()


def test_resolve_ckpt_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError):
        resolve_ckpt(tmp_path, "../../etc/passwd")


# --------------------------------------------------------------------------- #
# 2. 权重缺失 / 结构不兼容必须明确失败
# --------------------------------------------------------------------------- #
def test_load_model_missing_weight_raises(tmp_path):
    from tasks.goal5_segmentation.inference import load_model

    with pytest.raises(FileNotFoundError):
        load_model(Goal5Config(), tmp_path, device="cpu")


def test_load_model_rejects_incompatible_channels(tmp_path):
    torch = pytest.importorskip("torch")
    from tasks.goal5_segmentation.inference import load_model

    ckpt = tmp_path / "goal5_segmentation"
    ckpt.mkdir(parents=True)
    # in_ch 与配置不符（3 vs 4）→ 必须拒绝，否则会静默加载随机层
    torch.save({"arch": "mednext", "model_cfg": {"in_ch": 3}, "cls_spec": []},
               ckpt / "core.pt")
    with pytest.raises(ValueError):
        load_model(Goal5Config(in_channels=4), tmp_path, device="cpu")


# --------------------------------------------------------------------------- #
# 3. 结果类型契约
# --------------------------------------------------------------------------- #
def test_goal5_result_shape_and_dtype_contract():
    m = np.zeros((4, 5, 6), dtype=np.uint8)
    r = Goal5Result(core_mask=m, core_source_series_uid="uid_core",
                    flair_mask=m.copy(), flair_source_series_uid="uid_flair")
    assert r.core_mask.dtype == np.uint8
    assert set(np.unique(r.core_mask)) <= {0, 1}
    assert r.core_source_series_uid and r.flair_source_series_uid


# --------------------------------------------------------------------------- #
# 4. 比赛入口不导入训练专用模块（规范 §17.1）
# --------------------------------------------------------------------------- #
def test_entrypoint_does_not_import_training_modules():
    banned = {"dataset", "augmentations", "losses", "train", "evaluate"}
    offenders: list[str] = []
    for f in sorted((ROOT / "tasks" / "goal5_segmentation").rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for mod in mods:
                if "goal5_segmentation" in mod and mod.rsplit(".", 1)[-1] in banned:
                    offenders.append(f"{f.name}: {mod}")
    assert not offenders, f"比赛入口导入了训练专用模块: {offenders}"


# --------------------------------------------------------------------------- #
# 5. 插件不产生比赛目录副作用
# --------------------------------------------------------------------------- #
def test_task_does_not_touch_answer_dir(tmp_path, monkeypatch):
    """实例化与 load_model 失败都不应创建任何 answer/ 目录。"""
    from tasks.goal5_segmentation.task import Goal5Task

    monkeypatch.setenv("COMPETITION_WORKSPACE", str(tmp_path / "ws"))
    task = Goal5Task(settings=Settings.from_env())
    assert task.name == "goal5_segmentation"
    # 未加载权重时 predict 会尝试加载并因缺权重失败，但不得写任何文件
    study = Study(accession_number="ACC1", series=(
        Series(series_uid="S1", modality="t1c post contrast",
               image=np.ones((4, 4, 4), np.float32), affine=np.eye(4),
               source_path=tmp_path / "S1.nii.gz", metadata={}),
    ))
    from pipeline.context import PipelineContext
    with pytest.raises(FileNotFoundError):
        task.predict(PipelineContext(study=study))
    assert not (tmp_path / "ws" / "answer").exists()


# --------------------------------------------------------------------------- #
# 6. 缺模态必须走"零占位 + warning"，而不是崩溃
# --------------------------------------------------------------------------- #
def test_preprocess_fills_missing_channels(tmp_path):
    from tasks.goal5_segmentation.preprocess import CHANNEL_ORDER, build_volume

    study = Study(accession_number="ACC1", series=(
        Series(series_uid="S1", modality="t1c post contrast",
               image=np.random.RandomState(0).rand(8, 8, 8).astype(np.float32) * 100,
               affine=np.eye(4), source_path=tmp_path / "S1.nii.gz", metadata={}),
    ))
    prep = build_volume(study, Goal5Config())
    assert prep.volume.shape[0] == len(CHANNEL_ORDER)
    assert set(prep.missing) == {"flair", "t2", "t1"}


def test_study_rejects_empty_series_at_construction():
    """``Study`` 在**构造期**就拒绝空 series。

    这条由团队公共数据结构保证的契约很重要：它意味着"无影像的 Study"
    根本不可能进入 Pipeline，因此插件无需为它设计降级分支。
    （规范 §9.1 把"整个 Study 无有效影像"定义为不可降级错误，
     这里进一步说明它连构造都过不去。）
    """
    with pytest.raises(ValueError):
        Study(accession_number="ACC_EMPTY", series=())


def test_study_rejects_duplicate_series_uid(tmp_path):
    """同一 Study 内 series_uid 不得重复（规范 §6.1）。"""
    s = Series(series_uid="S1", modality="t1c", image=np.ones((4, 4, 4), np.float32),
               affine=np.eye(4), source_path=tmp_path / "a.nii.gz", metadata={})
    with pytest.raises(ValueError):
        Study(accession_number="ACC1", series=(s, s))
