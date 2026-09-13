from __future__ import annotations

import asyncio
from datetime import date
import json
import logging
from pathlib import Path
from uuid import UUID

from litestar import Request, patch, post
from litestar.datastructures import UploadFile
from litestar.exceptions import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.model_paths import ensure_smartbite_models
from app.ai.mobile_expiry_pipeline import MobileExpiryPipeline, MobileExpiryPipelineResult
from app.domain.repositories import DomainRepository, MobileExpiryScanCreateData
from app.domain.schemas import MobileExpiryCorrectionRequest, MobileExpiryScanResponse
from app.infra.settings import get_settings

logger = logging.getLogger(__name__)
console_logger = logging.getLogger("uvicorn.error")


def _parse_metadata(metadata_raw: object) -> dict | None:
    if metadata_raw is None or metadata_raw == "":
        return None
    try:
        if isinstance(metadata_raw, str):
            value = json.loads(metadata_raw)
        elif isinstance(metadata_raw, bytes):
            value = json.loads(metadata_raw.decode("utf-8"))
        else:
            raise TypeError("unsupported metadata type")
    except (TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="metadata must be a valid JSON object") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="metadata must be a valid JSON object")
    return value


def _clean_message(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _validate_upload(*, image_bytes: bytes, filename: str, content_type: str) -> None:
    settings = get_settings()
    if len(image_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=400, detail="file too large")
    if content_type not in settings.accepted_file_types:
        raise HTTPException(status_code=400, detail="unsupported content type")
    if Path(filename).suffix.lower() not in settings.accepted_file_extensions:
        raise HTTPException(status_code=400, detail="unsupported file extension")


def _response(row, *, pipeline_result: MobileExpiryPipelineResult | None = None) -> MobileExpiryScanResponse:
    return MobileExpiryScanResponse(
        id=row.id,
        status=row.status,
        expiry_date=row.corrected_expiry_date or row.detected_expiry_date,
        detected_expiry_date=row.detected_expiry_date,
        corrected_expiry_date=row.corrected_expiry_date,
        raw_text=row.raw_text,
        normalized_text=row.normalized_text,
        recognition_confidence=row.recognition_confidence,
        detector_confidence=row.detector_confidence,
        reason=row.reason,
        final_recognition_bbox_xyxy=(
            pipeline_result.final_recognition_bbox_xyxy if pipeline_result is not None else None
        ),
        final_recognition_polygon_json=(
            pipeline_result.final_recognition_polygon_json if pipeline_result is not None else None
        ),
        final_crop_policy=pipeline_result.final_crop_policy if pipeline_result is not None else None,
        final_crop_padding_px=pipeline_result.final_crop_padding_px if pipeline_result is not None else None,
        created_at=row.created_at,
    )


def _mobile_scan_result_log_payload(row, result: MobileExpiryPipelineResult) -> dict[str, object]:
    return {
        "scan_id": str(row.id),
        "status": row.status,
        "expiry_date": row.corrected_expiry_date.isoformat()
        if row.corrected_expiry_date
        else row.detected_expiry_date.isoformat()
        if row.detected_expiry_date
        else None,
        "detected_expiry_date": row.detected_expiry_date.isoformat() if row.detected_expiry_date else None,
        "raw_text": row.raw_text,
        "normalized_text": row.normalized_text,
        "recognition_confidence": row.recognition_confidence,
        "detector_confidence": row.detector_confidence,
        "reason": row.reason,
        "elapsed_ms": result.runtime_ms,
    }


def _log_mobile_scan_result(row, result: MobileExpiryPipelineResult) -> None:
    payload = _mobile_scan_result_log_payload(row, result)
    logger.info(
        "mobile expiry scan completed",
        extra=payload,
    )
    console_logger.info(
        "mobile expiry scan completed %s",
        json.dumps(payload, sort_keys=True),
    )


async def _get_mobile_pipeline(request: Request) -> MobileExpiryPipeline:
    existing = getattr(request.app.state, "mobile_expiry_pipeline", None)
    if isinstance(existing, MobileExpiryPipeline):
        return existing

    lock = getattr(request.app.state, "mobile_expiry_pipeline_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.mobile_expiry_pipeline_lock = lock

    async with lock:
        existing = getattr(request.app.state, "mobile_expiry_pipeline", None)
        if isinstance(existing, MobileExpiryPipeline):
            return existing
        settings = get_settings()
        if settings.models_auto_download:
            await asyncio.to_thread(
                ensure_smartbite_models,
                repo_id=settings.models_hf_repo_id,
                revision=settings.models_hf_revision,
                detector_path=settings.mobile_expiry_detector_model_path,
                detector_onnx_path=settings.mobile_expiry_detector_onnx_path,
                svtr_model_dir=settings.svtrv2_rec_model_dir,
                svtr_onnx_path=settings.svtrv2_rec_onnx_path,
            )
        pipeline = MobileExpiryPipeline(
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
            rapidocr_primary_mode=settings.mobile_rapidocr_primary_mode,
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
            rapidocr_gated_max_boxes_accepted=settings.mobile_rapidocr_gated_max_boxes_accepted,
            rapidocr_gated_probe_top_k=settings.mobile_rapidocr_gated_probe_top_k,
            rapidocr_gated_final_top_k=settings.mobile_rapidocr_gated_final_top_k,
            rapidocr_gated_prefilter_probe_top_k=settings.mobile_rapidocr_gated_prefilter_probe_top_k,
            rapidocr_gated_min_probe_digits=settings.mobile_rapidocr_gated_min_probe_digits,
            rapidocr_gated_min_date_likeness_score=settings.mobile_rapidocr_gated_min_date_likeness_score,
            rapidocr_max_rois_per_scan=settings.mobile_rapidocr_max_rois_per_scan,
            rapidocr_max_boxes_accepted=settings.mobile_rapidocr_max_boxes_accepted,
            rapidocr_timeout_seconds=settings.mobile_rapidocr_timeout_seconds,
            rapidocr_min_confidence=settings.mobile_rapidocr_min_confidence,
            product_cropper_rescue_enabled=settings.mobile_product_cropper_rescue_enabled,
            product_cropper_model_path=settings.mobile_product_cropper_model_path,
            product_cropper_confidence_threshold=settings.mobile_product_cropper_confidence_threshold,
            product_cropper_imgsz=settings.mobile_product_cropper_imgsz,
            product_cropper_padding_ratio=settings.mobile_product_cropper_padding_ratio,
            product_cropper_max_candidates=settings.mobile_product_cropper_max_candidates,
            legacy_wide_group_rescue_enabled=settings.mobile_legacy_wide_group_rescue_enabled,
            strict_evidence_acceptance_enabled=settings.mobile_strict_evidence_acceptance_enabled,
            rapidocr_role_constraint_enabled=settings.mobile_rapidocr_role_constraint_enabled,
        )
        request.app.state.mobile_expiry_pipeline = pipeline
        return pipeline


@post("/mobile/expiry-scans", status_code=201)
async def create_mobile_expiry_scan(request: Request, session: AsyncSession) -> MobileExpiryScanResponse:
    form = await request.form()
    image = form.get("image")
    if not isinstance(image, UploadFile):
        raise HTTPException(status_code=400, detail="image is required")

    filename = image.filename or "upload.jpg"
    content_type = image.content_type or "application/octet-stream"
    image_bytes = await image.read()
    _validate_upload(image_bytes=image_bytes, filename=filename, content_type=content_type)

    pipeline = await _get_mobile_pipeline(request)
    result = await asyncio.to_thread(pipeline.run, image_bytes, today=date.today())

    repo = DomainRepository(session)
    row = await repo.create_mobile_expiry_scan(
        MobileExpiryScanCreateData(
            message=_clean_message(form.get("message")),
            metadata_json=_parse_metadata(form.get("metadata")),
            image_blob=image_bytes,
            image_filename=Path(filename).name,
            image_content_type=content_type,
            detected_expiry_date=result.detected_expiry_date,
            raw_text=result.raw_text,
            normalized_text=result.normalized_text,
            recognition_confidence=result.recognition_confidence,
            detector_confidence=result.detector_confidence,
            status=result.status,
            reason=result.reason,
            detection_polygon_json=result.detection_polygon_json,
        )
    )
    await session.commit()
    _log_mobile_scan_result(row, result)
    return _response(row, pipeline_result=result)


@patch("/mobile/expiry-scans/{scan_id:uuid}")
async def correct_mobile_expiry_scan(
    scan_id: UUID,
    data: MobileExpiryCorrectionRequest,
    session: AsyncSession,
) -> MobileExpiryScanResponse:
    repo = DomainRepository(session)
    row = await repo.get_mobile_expiry_scan(scan_id)
    if row is None:
        raise HTTPException(status_code=404, detail="mobile expiry scan not found")

    corrected = await repo.correct_mobile_expiry_scan(
        row,
        corrected_expiry_date=data.corrected_expiry_date,
        reason=data.reason.strip() if isinstance(data.reason, str) and data.reason.strip() else None,
    )
    await session.commit()
    return _response(corrected)
