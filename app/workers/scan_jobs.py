from __future__ import annotations

import logging
from datetime import date
from uuid import UUID

from arq.connections import RedisSettings

from app.ai.decision import ExpiryDecisionEngine
from app.ai.detector import ExpiryRegionDetector
from app.ai.ocr import OCRConfig, OCRRouter
from app.ai.parser import ExpiryDateParser
from app.ai.pipeline import ExpiryPipeline
from app.ai.preprocess import ROIImagePreprocessor
from app.domain.enums import AlertType, ExpiryClassification, FinalResultStatus, ScanStatus
from app.domain.repositories import DomainRepository
from app.infra.db import get_session_factory
from app.infra.settings import Settings, get_settings
from app.infra.storage import LocalStorage

logger = logging.getLogger(__name__)


def build_pipeline(settings: Settings) -> ExpiryPipeline:
    detector = ExpiryRegionDetector(settings.detector_model_path)
    preprocessor = ROIImagePreprocessor()
    ocr_router = OCRRouter(
        OCRConfig(
            onnx_model_path=settings.ocr_onnx_model_path,
            onnx_charset_path=settings.ocr_onnx_charset_path,
            paddle_lang=settings.ocr_paddle_lang,
            device_mode=settings.ocr_device_mode,
        )
    )
    parser = ExpiryDateParser()
    decision = ExpiryDecisionEngine(alert_threshold_days=settings.alert_threshold_days)
    return ExpiryPipeline(detector, preprocessor, ocr_router, parser, decision)


async def startup(ctx: dict) -> None:
    settings = get_settings()
    ctx["settings"] = settings
    ctx["storage"] = LocalStorage(settings.storage_root)
    ctx["pipeline"] = build_pipeline(settings)


async def process_scan(ctx: dict, scan_id: str) -> None:
    storage: LocalStorage = ctx["storage"]
    pipeline: ExpiryPipeline = ctx["pipeline"]

    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = DomainRepository(session)
        scan = await repo.get_scan(UUID(scan_id))
        if scan is None:
            return
        if scan.status == ScanStatus.DONE:
            return

        await repo.set_scan_processing(scan)
        await repo.add_log(scan_id=scan.id, stage="worker", message="scan processing started")
        await session.commit()

        image_bytes = storage.read_bytes(scan.image_path)
        output = pipeline.run(image_bytes, today=date.today())

        roi_path = scan.roi_path
        if output.roi_png_bytes is not None:
            roi_path = storage.roi_relative_path(scan.id)
            storage.save_bytes(roi_path, output.roi_png_bytes)

        if output.raw_text is not None:
            await repo.upsert_ocr_result(
                scan_id=scan.id,
                raw_text=output.raw_text,
                normalized_text=output.normalized_text or "",
                ocr_confidence=output.ocr_confidence,
                engine_name=output.ocr_engine or "unknown",
                runtime_device=output.ocr_runtime_device,
                reason=output.reason if output.final_status == FinalResultStatus.OCR_FAILED.value else None,
            )

        await repo.upsert_parsed_result(
            scan_id=scan.id,
            parsed_date=output.parsed_date,
            date_format_detected=output.date_format_detected,
            parse_confidence=output.parse_confidence,
            parser_reason=output.reason,
            candidates=[output.parsed_date.isoformat()] if output.parsed_date else [],
        )

        await repo.upsert_expiry_status(
            scan_id=scan.id,
            status=ExpiryClassification(output.expiry_classification),
            days_remaining=output.days_remaining,
            alert_required=output.alert_required,
            needs_review=output.needs_review,
            reason=output.reason,
        )

        await repo.finalize_scan(
            scan,
            final_status=FinalResultStatus(output.final_status),
            detected=output.detected,
            detector_confidence=output.detector_confidence,
            failure_reason=output.reason,
            roi_path=roi_path,
        )

        for stage, elapsed in output.stage_timings_ms.items():
            await repo.add_log(
                scan_id=scan.id,
                stage=stage,
                message=f"stage completed: {stage}",
                elapsed_ms=elapsed,
                payload={
                    "detector_confidence": output.detector_confidence,
                    "ocr_confidence": output.ocr_confidence,
                    "final_status": output.final_status,
                    "classification": output.expiry_classification,
                },
            )

        if output.alert_required:
            alert_type = AlertType.MANUAL_REVIEW
            if output.expiry_classification == ExpiryClassification.EXPIRED:
                alert_type = AlertType.EXPIRED
            elif output.expiry_classification == ExpiryClassification.EXPIRING_SOON:
                alert_type = AlertType.EXPIRING_SOON

            await repo.create_alert_if_missing(
                scan_id=scan.id,
                user_id=scan.user_id,
                alert_type=alert_type,
                payload={
                    "scan_id": str(scan.id),
                    "user_id": scan.user_id,
                    "classification": output.expiry_classification,
                    "days_remaining": output.days_remaining,
                    "reason": output.reason,
                },
            )

        await session.commit()


class WorkerSettings:
    functions = [process_scan]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    retry_jobs = True
