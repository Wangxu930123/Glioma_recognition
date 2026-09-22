from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.callback import CompetitionCallback
from core.config import Settings
from core.registry import build_pipeline
from data.loader import DatasetLoader
from observability.competition_logger import CompetitionLogger
from output.validator import OutputValidator
from output.writer import OutputWriter
from pipeline.inference import InferencePipeline


@dataclass(frozen=True)
class EvaluationJob:
    request_id: str
    evaluation_id: str
    dataset_path: Path


class EvaluationRunner:
    def __init__(
        self,
        settings: Settings,
        *,
        loader: DatasetLoader | None = None,
        pipeline: InferencePipeline | None = None,
        validator: OutputValidator | None = None,
        callback: CompetitionCallback | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader or DatasetLoader()
        self.pipeline = pipeline or build_pipeline(settings.pipeline_factory)
        self.writer = OutputWriter(settings.answer_root)
        self.validator = validator or OutputValidator()
        self.callback = callback or CompetitionCallback(
            settings.callback_url,
            settings.callback_timeout_seconds,
            settings.callback_attempts,
        )
        self.logger = CompetitionLogger(settings.log_root)
        self._run_lock = threading.Lock()

    def run(self, job: EvaluationJob, *, send_callback: bool = True) -> Path:
        # DatasetTask has one mutable incremental state, so evaluations sharing
        # this runner must not interleave even if the job executor has workers.
        with self._run_lock:
            return self._run_streaming(job, send_callback=send_callback)

    def _run_streaming(
        self,
        job: EvaluationJob,
        *,
        send_callback: bool,
    ) -> Path:
        started = time.perf_counter()
        data_source = str(job.dataset_path)
        self.logger.write(
            request_id=job.request_id,
            evaluation_id=job.evaluation_id,
            phase="test",
            message="evaluation_started",
            data_source=data_source,
        )
        staging: Path | None = None
        current_accession: str | None = None
        try:
            staging = self.writer.begin(job.evaluation_id)
            accessions: set[str] = set()
            self.pipeline.reset_dataset_task()
            for study in self.loader.iter_studies(job.dataset_path):
                current_accession = study.accession_number
                if current_accession in accessions:
                    raise ValueError(
                        f"dataset has duplicate accession number: {current_accession}"
                    )
                context = self.pipeline.run_study(study)
                self.pipeline.update_dataset_task(study, context)
                accession_dir = self.writer.write_study(staging, context)
                self.validator.validate_study(accession_dir, study)
                accessions.add(current_accession)
                del context, study
                current_accession = None

            duplicates = self.pipeline.finalize_dataset_task()
            duplicate_path = self.writer.write_duplicates(
                staging,
                accessions,
                duplicates,
            )
            self.validator.validate_duplicates(duplicate_path, accessions)
            self.validator.validate_final_layout(staging, accessions)
            output_dir = self.writer.publish(staging, job.evaluation_id)
            staging = None
            duration_ms = round((time.perf_counter() - started) * 1000)
            self.logger.write(
                request_id=job.request_id,
                evaluation_id=job.evaluation_id,
                phase="test",
                message="evaluation_completed",
                data_source=data_source,
                duration_ms=duration_ms,
                study_count=len(accessions),
                pred_path=str(output_dir),
            )
            if send_callback:
                self.callback.send_success(
                    job.request_id,
                    job.evaluation_id,
                    output_dir,
                )
            return output_dir
        except Exception as exc:
            self.logger.write(
                request_id=job.request_id,
                evaluation_id=job.evaluation_id,
                phase="test",
                message="evaluation_failed",
                data_source=data_source,
                error_type=type(exc).__name__,
                error=str(exc),
                accession_number=current_accession,
            )
            if staging and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise
