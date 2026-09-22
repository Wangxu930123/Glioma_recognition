from __future__ import annotations

import importlib

from pipeline.inference import InferencePipeline


def build_pipeline(factory_path: str | None) -> InferencePipeline:
    """Load ``module:function`` when real task plugins replace the baseline."""
    if not factory_path:
        return InferencePipeline()
    module_name, separator, function_name = factory_path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "COMPETITION_PIPELINE_FACTORY must use the form ''module:function''"
        )
    factory = getattr(importlib.import_module(module_name), function_name)
    pipeline = factory()
    if not isinstance(pipeline, InferencePipeline):
        raise TypeError(f"{factory_path} did not return InferencePipeline")
    return pipeline
