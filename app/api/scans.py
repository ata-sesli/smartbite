from __future__ import annotations

import json
from uuid import UUID

from litestar import Request, get, patch, post
from litestar.datastructures import UploadFile
from litestar.exceptions import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas import ManualCorrectionRequest, ManualCorrectionResponse, ScanCreateResponse, ScanGetResponse
from app.domain.services import ScanService, ValidationError
from app.infra.settings import get_settings
from app.infra.storage import LocalStorage


def _scan_service(request: Request, session: AsyncSession) -> ScanService:
    settings = get_settings()
    storage = LocalStorage(settings.storage_root)
    redis = getattr(request.app.state, "redis", None)
    return ScanService(
        session,
        storage,
        redis,
        max_upload_size_bytes=settings.max_upload_size_bytes,
        accepted_file_types=settings.accepted_file_types,
        accepted_file_extensions=settings.accepted_file_extensions,
        alert_threshold_days=settings.alert_threshold_days,
    )


@post("/scans")
async def create_scan(request: Request, session: AsyncSession) -> ScanCreateResponse:
    form = await request.form()
    image = form.get("image")
    qr_code = str(form.get("qr_code", "")).strip()
    user_id = str(form.get("user_id", "")).strip()
    metadata_raw = form.get("metadata")

    if not isinstance(image, UploadFile):
        raise HTTPException(status_code=400, detail="image is required")
    if not qr_code:
        raise HTTPException(status_code=400, detail="qr_code is required")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    metadata = None
    if metadata_raw:
        try:
            if isinstance(metadata_raw, str):
                metadata = json.loads(metadata_raw)
            elif isinstance(metadata_raw, bytes):
                metadata = json.loads(metadata_raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc

    payload = await image.read()

    service = _scan_service(request, session)
    try:
        scan_id = await service.create_scan(
            image_bytes=payload,
            filename=image.filename or "upload.jpg",
            content_type=image.content_type or "application/octet-stream",
            qr_code=qr_code,
            user_id=user_id,
            metadata=metadata,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ScanCreateResponse(scan_id=scan_id, status="queued")


@get("/scans/{scan_id:uuid}")
async def get_scan(request: Request, session: AsyncSession, scan_id: UUID) -> ScanGetResponse:
    service = _scan_service(request, session)
    try:
        status, result = await service.get_scan_payload(scan_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ScanGetResponse(scan_id=scan_id, status=status, result=result)


@patch("/scans/{scan_id:uuid}")
async def patch_scan(
    request: Request,
    session: AsyncSession,
    scan_id: UUID,
    data: ManualCorrectionRequest,
) -> ManualCorrectionResponse:
    service = _scan_service(request, session)
    try:
        status, final_status, classification, days_remaining = await service.manual_correct_scan(
            scan_id, parsed_date=data.parsed_date, reason=data.reason
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ManualCorrectionResponse(
        scan_id=scan_id,
        status=status,
        final_status=final_status,
        expiry_classification=classification,
        days_remaining=days_remaining,
    )
