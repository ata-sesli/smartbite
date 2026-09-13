from __future__ import annotations

from uuid import UUID

from litestar import Request, get, post, put
from litestar.exceptions import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas import (
    ScanReviewResponse,
    ScanReviewUpsertRequest,
    TrainingDataExportRequest,
    TrainingDataExportResponse,
)
from app.domain.services import ScanService, ValidationError
from app.infra.settings import get_settings
from app.infra.storage import LocalStorage


def _scan_service(request: Request, session: AsyncSession) -> ScanService:
    settings = get_settings()
    return ScanService(
        session,
        LocalStorage(settings.storage_root),
        getattr(request.app.state, "redis", None),
        max_upload_size_bytes=settings.max_upload_size_bytes,
        accepted_file_types=settings.accepted_file_types,
        accepted_file_extensions=settings.accepted_file_extensions,
        alert_threshold_days=settings.alert_threshold_days,
        scan_job_expires_seconds=settings.scan_job_expires_seconds,
        general_text_ocr_timeout_seconds=settings.general_text_ocr_timeout_seconds,
    )


@get("/scans/{scan_id:uuid}/review")
async def get_scan_review(request: Request, session: AsyncSession, scan_id: UUID) -> ScanReviewResponse:
    service = _scan_service(request, session)
    try:
        review = await service.get_scan_review_payload(scan_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ScanReviewResponse(scan_id=scan_id, review=review)


@put("/scans/{scan_id:uuid}/review")
async def put_scan_review(
    request: Request,
    session: AsyncSession,
    scan_id: UUID,
    data: ScanReviewUpsertRequest,
) -> ScanReviewResponse:
    service = _scan_service(request, session)
    try:
        review = await service.upsert_scan_review(
            scan_id=scan_id,
            verdict=data.verdict,
            accepted_for_training=data.accepted_for_training,
            final_text=data.final_text,
            final_parsed_date=data.final_parsed_date,
            bbox_xyxy=data.bbox_xyxy,
            bbox_source=data.bbox_source,
            reviewer_id=data.reviewer_id,
            notes=data.notes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ScanReviewResponse(scan_id=scan_id, review=review)


@post("/training-data/export")
async def export_training_data(
    request: Request,
    session: AsyncSession,
    data: TrainingDataExportRequest,
) -> TrainingDataExportResponse:
    service = _scan_service(request, session)
    return await service.export_training_data(
        output_dir=data.output_dir,
        include_detector=data.include_detector,
        include_recognition=data.include_recognition,
    )
