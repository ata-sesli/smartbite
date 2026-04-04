from __future__ import annotations

import importlib
import logging
import os
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


def _assert_worker_ai_dependencies() -> None:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    required_modules = ("ultralytics", "paddle", "paddleocr")
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


def _log_recognition_dict_alignment(settings: Settings) -> None:
    inference_yml = settings.ocr_ppocrv5_main_model_dir / "inference.yml"
    char_dict = settings.ocr_ppocrv5_main_char_dict_path
    if not inference_yml.exists() or char_dict is None or not char_dict.exists():
        return

    try:
        import yaml

        payload = yaml.safe_load(inference_yml.read_text()) or {}
        embedded_dict = payload.get("PostProcess", {}).get("character_dict")
        if not isinstance(embedded_dict, list):
            return
        file_dict = [line.rstrip("\n") for line in char_dict.read_text().splitlines()]
        if embedded_dict == file_dict:
            logger.info("OCR dict alignment OK: embedded inference dict matches %s", char_dict)
        else:
            logger.warning(
                "OCR dict mismatch: embedded inference dict differs from %s (embedded=%s file=%s)",
                char_dict,
                len(embedded_dict),
                len(file_dict),
            )
    except Exception as exc:
        logger.warning("Could not validate OCR dict alignment: %s", exc)


def build_pipeline(settings: Settings) -> ExpiryPipeline:
    detector = ExpiryRegionDetector(settings.detector_model_path)
    preprocessor = ROIImagePreprocessor()
    ocr_router = OCRRouter(
        OCRConfig(
            ppocrv5_main_model_dir=settings.ocr_ppocrv5_main_model_dir,
            ppocrv5_main_char_dict_path=settings.ocr_ppocrv5_main_char_dict_path,
            paddle_lang=settings.ocr_paddle_lang,
            device_mode=settings.ocr_device_mode,
            ppocrv5_use_angle_cls=settings.ocr_ppocrv5_use_angle_cls,
            ppocrv5_det_db_thresh=settings.ocr_ppocrv5_det_db_thresh,
            ppocrv5_det_db_box_thresh=settings.ocr_ppocrv5_det_db_box_thresh,
            substitute_config_path=settings.ocr_substitute_config_path,
            enable_substitute_model=settings.ocr_enable_substitute_model,
        )
    )
    parser = ExpiryDateParser()
    decision = ExpiryDecisionEngine(alert_threshold_days=settings.alert_threshold_days)
    return ExpiryPipeline(detector, preprocessor, ocr_router, parser, decision)


async def startup(ctx: dict) -> None:
    _assert_worker_ai_dependencies()
    settings = get_settings()
    _log_recognition_dict_alignment(settings)
    logger.info(
        "OCR startup config: main_model_dir=%s char_dict=%s device_mode=%s det_thresh=%s box_thresh=%s substitute_enabled=%s",
        settings.ocr_ppocrv5_main_model_dir,
        settings.ocr_ppocrv5_main_char_dict_path,
        settings.ocr_device_mode,
        settings.ocr_ppocrv5_det_db_thresh,
        settings.ocr_ppocrv5_det_db_box_thresh,
        settings.ocr_enable_substitute_model,
    )
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
