from __future__ import annotations

import asyncio
import importlib
import logging
import os
from datetime import date
from uuid import UUID

from arq.connections import RedisSettings

from app.ai.decision import ExpiryDecisionEngine
from app.ai.mobile_expiry_pipeline import MobileExpiryPipeline, MobileExpiryPipelineResult
from app.domain.enums import AlertType, ExpiryClassification, FinalResultStatus, ScanStatus
from app.domain.repositories import DomainRepository
from app.infra.db import get_session_factory
from app.infra.settings import Settings, get_settings
from app.infra.storage import LocalStorage

logger = logging.getLogger(__name__)


def _assert_worker_ai_dependencies(settings: Settings) -> None:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    required_modules = ["ultralytics", "paddle", "paddleocr", "torch"]
    if settings.mobile_expiry_detector_backend == "onnx" or settings.svtrv2_rec_backend == "onnx":
        required_modules.append("onnxruntime")
    if (
        settings.mobile_proposal_rescue_backend == "rapidocr_ppocrv5"
        and (settings.mobile_rapidocr_primary_enabled or settings.mobile_rapidocr_rescue_enabled)
    ):
        required_modules.append("rapidocr")
    missing: list[str] = []

    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - import environment specific
            missing.append(f"{module_name} ({exc})")

    if missing:
        raise RuntimeError(
            "worker AI dependencies are missing. "
            "Run `uv sync --extra ai` (preferred) or `pip install -e '.[ai]'` before starting smartbite-worker. "
            f"Missing modules: {', '.join(missing)}"
        )


def build_pipeline(settings: Settings) -> MobileExpiryPipeline:
    return MobileExpiryPipeline(
        detector_model_path=settings.mobile_expiry_detector_model_path,
        detector_confidence_threshold=settings.mobile_expiry_detector_confidence_threshold,
        detector_imgsz=settings.mobile_expiry_detector_imgsz,
        max_candidates=settings.mobile_expiry_max_candidates,
        crop_padding_px=settings.mobile_expiry_crop_padding_px,
        svtr_model_name=settings.svtrv2_rec_model_name,
        svtr_model_dir=settings.svtrv2_rec_model_dir,
        svtr_device=settings.svtrv2_device_mode,
        parser_min_candidate_confidence=settings.parser_min_candidate_confidence,
        detector_backend=settings.mobile_expiry_detector_backend,
        detector_onnx_path=settings.mobile_expiry_detector_onnx_path,
        svtr_backend=settings.svtrv2_rec_backend,
        svtr_onnx_model_path=settings.svtrv2_rec_onnx_path,
        proposal_rescue_backend=settings.mobile_proposal_rescue_backend,
        rapidocr_primary_enabled=settings.mobile_rapidocr_primary_enabled,
        rapidocr_rescue_enabled=settings.mobile_rapidocr_rescue_enabled,
        rapidocr_ocr_version=settings.mobile_rapidocr_ocr_version,
        rapidocr_model_type=settings.mobile_rapidocr_model_type,
        rapidocr_lang_type=settings.mobile_rapidocr_lang_type,
        rapidocr_limit_side_len=settings.mobile_rapidocr_limit_side_len,
        rapidocr_limit_type=settings.mobile_rapidocr_limit_type,
        rapidocr_max_candidates=settings.mobile_rapidocr_max_candidates,
        rapidocr_primary_max_rois_per_scan=settings.mobile_rapidocr_primary_max_rois_per_scan,
        rapidocr_primary_max_boxes_accepted=settings.mobile_rapidocr_primary_max_boxes_accepted,
        rapidocr_primary_timeout_seconds=settings.mobile_rapidocr_primary_timeout_seconds,
        rapidocr_max_rois_per_scan=settings.mobile_rapidocr_max_rois_per_scan,
        rapidocr_max_boxes_accepted=settings.mobile_rapidocr_max_boxes_accepted,
        rapidocr_timeout_seconds=settings.mobile_rapidocr_timeout_seconds,
        rapidocr_min_confidence=settings.mobile_rapidocr_min_confidence,
    )


async def startup(ctx: dict) -> None:
    settings = get_settings()
    _assert_worker_ai_dependencies(settings)
    logger.info(
        "expiry worker startup: detector_backend=%s detector_model=%s detector_onnx=%s detector_conf=%s imgsz=%s max_candidates=%s recognizer=svtrv2 recognizer_backend=%s recognizer_model=%s recognizer_dir=%s recognizer_onnx=%s recognizer_device=%s parser_min_conf=%s",
        settings.mobile_expiry_detector_backend,
        settings.mobile_expiry_detector_model_path,
        settings.mobile_expiry_detector_onnx_path,
        settings.mobile_expiry_detector_confidence_threshold,
        settings.mobile_expiry_detector_imgsz,
        settings.mobile_expiry_max_candidates,
        settings.svtrv2_rec_backend,
        settings.svtrv2_rec_model_name,
        settings.svtrv2_rec_model_dir,
        settings.svtrv2_rec_onnx_path,
        settings.svtrv2_device_mode,
        settings.parser_min_candidate_confidence,
    )
    ctx["settings"] = settings
    ctx["storage"] = LocalStorage(settings.storage_root)
    ctx["pipeline"] = build_pipeline(settings)
    ctx["decision"] = ExpiryDecisionEngine(alert_threshold_days=settings.alert_threshold_days)


def _final_status_for_output(output: MobileExpiryPipelineResult, decision_status: FinalResultStatus) -> FinalResultStatus:
    if output.status == "parsed_success":
        return decision_status
    if output.status == "failed":
        return FinalResultStatus.OCR_FAILED
    return FinalResultStatus.MANUAL_REVIEW_REQUIRED


async def process_scan(ctx: dict, scan_id: str) -> None:
    storage: LocalStorage = ctx["storage"]
    pipeline: MobileExpiryPipeline = ctx["pipeline"]
    decision_engine: ExpiryDecisionEngine = ctx["decision"]
    scan_uuid = UUID(scan_id)

    session_factory = get_session_factory()

    async with session_factory() as session:
        repo = DomainRepository(session)
        scan = await repo.get_scan(scan_uuid)
        if scan is None:
            return
        if scan.status == ScanStatus.DONE:
            return

        image_path = scan.image_path
        await repo.set_scan_processing(scan)
        await repo.add_log(scan_id=scan.id, stage="worker", message="scan processing started")
        await session.commit()

    image_bytes = storage.read_bytes(image_path)
    today = date.today()
    output = await asyncio.to_thread(pipeline.run, image_bytes, today=today)
    parse_confidence = output.recognition_confidence if output.recognition_confidence is not None else 0.0
    decision = decision_engine.decide(output.detected_expiry_date, today=today, parse_confidence=parse_confidence)
    final_status = _final_status_for_output(output, decision.final_status)

    async with session_factory() as session:
        repo = DomainRepository(session)
        scan = await repo.get_scan(scan_uuid)
        if scan is None:
            return
        if scan.status == ScanStatus.DONE:
            return

        if output.raw_text is not None:
            await repo.upsert_ocr_result(
                scan_id=scan.id,
                raw_text=output.raw_text,
                normalized_text=output.normalized_text or "",
                ocr_confidence=output.recognition_confidence,
                engine_name="svtrv2",
                runtime_device=get_settings().svtrv2_device_mode,
                reason=output.reason if final_status != FinalResultStatus.PARSED_SUCCESS else None,
            )

        await repo.upsert_parsed_result(
            scan_id=scan.id,
            parsed_date=output.detected_expiry_date,
            date_format_detected="mobile_expiry_pipeline" if output.detected_expiry_date else None,
            parse_confidence=parse_confidence if output.detected_expiry_date else None,
            parser_reason=output.reason or decision.reason,
            candidates=[output.detected_expiry_date.isoformat()] if output.detected_expiry_date else [],
        )

        await repo.upsert_expiry_status(
            scan_id=scan.id,
            status=decision.expiry_classification,
            days_remaining=decision.days_remaining,
            alert_required=decision.alert_required,
            needs_review=decision.needs_review,
            reason=output.reason or decision.reason,
        )

        await repo.finalize_scan(
            scan,
            final_status=final_status,
            detected=output.detector_confidence is not None,
            detector_confidence=output.detector_confidence,
            failure_reason=output.reason,
            roi_path=scan.roi_path,
        )

        await repo.add_log(
            scan_id=scan.id,
            stage="mobile_expiry_pipeline",
            message="stage completed: mobile_expiry_pipeline",
            elapsed_ms=output.runtime_ms,
            payload={
                "detector_confidence": output.detector_confidence,
                "recognition_confidence": output.recognition_confidence,
                "final_status": final_status,
                "classification": decision.expiry_classification,
                "reason": output.reason,
                "engine_name": "svtrv2",
                "detector_model": str(get_settings().mobile_expiry_detector_model_path),
                "detection_polygon": output.detection_polygon_json,
            },
        )

        if decision.alert_required:
            alert_type = AlertType.MANUAL_REVIEW
            if decision.expiry_classification == ExpiryClassification.EXPIRED:
                alert_type = AlertType.EXPIRED
            elif decision.expiry_classification == ExpiryClassification.EXPIRING_SOON:
                alert_type = AlertType.EXPIRING_SOON

            await repo.create_alert_if_missing(
                scan_id=scan.id,
                user_id=scan.user_id,
                alert_type=alert_type,
                payload={
                    "scan_id": str(scan.id),
                    "user_id": scan.user_id,
                    "classification": decision.expiry_classification,
                    "days_remaining": decision.days_remaining,
                    "reason": output.reason or decision.reason,
                },
            )

        await session.commit()


class WorkerSettings:
    functions = [process_scan]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    retry_jobs = True
