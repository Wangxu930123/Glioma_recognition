from __future__ import annotations

from core.exceptions import InvalidTaskResultError
from pipeline.context import PipelineContext
from tasks.results import BinaryResult, CategoricalResult

#: 掩码 URI 模板（【待确认】组委会最终规则；变更只需改这一处，见 `_mask_uri`）
MASK_URI_TEMPLATE = "./{uid}/{uid}.nii.gz"


class PredictionAggregator:
    def build(self, context: PipelineContext) -> dict[str, object]:
        if not all(
            (
                context.goal1,
                context.goal2_stitched,
                context.goal3,
                context.goal4,
                context.goal5,
            )
        ):
            raise InvalidTaskResultError(
                f"incomplete results for {context.study.accession_number}"
            )

        goal4 = context.goal4
        goal5 = context.goal5
        prediction: dict[str, object] = {
            "AccessionNumber": context.study.accession_number,
            "IsNotHumanBodyProb": context.goal1.not_human_probability,
            "IsStitchedProb": context.goal2_stitched.stitched_probability,
            "ProcessingTime_ms": context.processing_time_ms,
            "SegmentationMaskURI": {
                "core": self._mask_uri(goal5.core_source_series_uid),
                "flair": self._mask_uri(goal5.flair_source_series_uid),
            },
            "Prediction": {
                "TumorProbability": context.goal3.tumor_probability,
                "Location": goal4.location,
                "Morphology": self._category(goal4.morphology),
                "WHO_Grade": self._category(goal4.who_grade),
                "Enhancement": self._binary(
                    goal4.enhancement,
                    "EnhancementProbability",
                ),
                "EnhancementPattern": self._category(goal4.enhancement_pattern),
                "Necrosis": self._binary(goal4.necrosis, "NecrosisProbability"),
                "CysticChange": self._binary(
                    goal4.cystic_change,
                    "CysticChangeProbability",
                ),
                "Hemorrhage": self._binary(
                    goal4.hemorrhage,
                    "HemorrhageProbability",
                ),
                "Calcification": self._binary(
                    goal4.calcification,
                    "CalcificationProbability",
                ),
                "Margin": {
                    "clear": goal4.margin_clear.present,
                    "MarginClearProbability": goal4.margin_clear.probability,
                },
                "Lobulation": self._binary(
                    goal4.lobulation,
                    "LobulationProbability",
                ),
                "Signal_T2WI": self._category(goal4.signal_t2wi),
                "Signal_FLAIR": self._category(goal4.signal_flair),
            },
            "Interpretation": {"Conclusion": goal4.conclusion},
        }
        if goal4.attention_map_uri:
            prediction["Interpretation"]["AttentionMapURI"] = goal4.attention_map_uri
        return prediction

    @staticmethod
    def _mask_uri(series_uid: str) -> str:
        """掩码 URI（**单点维护**）。

        【待确认】组委会材料中"目录规范"与"阳性示例"对掩码路径的描述不一致：
        目录规范为 ``{AccessionNumber}/{SeriesUid}/<series>.nii.gz``，
        示例却写作 ``./output/sub-*_core.nii.gz``。

        当前实现采用 **相对 URI + ``{series_uid}`` 目录 + 同名文件**，与
        ``OutputWriter._write_masks`` 的物理落盘位置严格一致（``OutputValidator``
        会逐例校验 URI 指向真实文件、shape 精确相同、affine 在容差内）。
        这样在整个 evaluation 内是**自洽**的——即便组委会最终规则不同，
        也只需修改此处模板一处，Writer/Validator 会同步跟随。
        """
        return MASK_URI_TEMPLATE.format(uid=series_uid)

    @staticmethod
    def _category(result: CategoricalResult) -> dict[str, object]:
        return {
            "predicted": result.predicted,
            "probabilities": result.probabilities,
        }

    @staticmethod
    def _binary(result: BinaryResult, probability_key: str) -> dict[str, object]:
        return {
            "present": result.present,
            probability_key: result.probability,
        }

