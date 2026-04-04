from __future__ import annotations

import asyncio
import json
from time import monotonic
from uuid import UUID
from uuid import uuid4

from litestar import Request, get, patch, post
from litestar.datastructures import UploadFile
from litestar.exceptions import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import ScanStatus
from app.domain.schemas import (
    ManualCorrectionRequest,
    ManualCorrectionResponse,
    OneShotAnalyzeResponse,
    ScanCreateResponse,
    ScanGetResponse,
)
from app.domain.services import ScanService, ValidationError
from app.infra.db import get_session_factory
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


def _parse_metadata(metadata_raw: object) -> dict | None:
    metadata = None
    if metadata_raw:
        try:
            if isinstance(metadata_raw, str):
                metadata = json.loads(metadata_raw)
            elif isinstance(metadata_raw, bytes):
                metadata = json.loads(metadata_raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
    return metadata


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

    metadata = _parse_metadata(metadata_raw)

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


@post("/scans/oneshot", status_code=200)
async def create_one_shot_scan(request: Request, session: AsyncSession) -> OneShotAnalyzeResponse:
    form = await request.form()
    image = form.get("image")
    metadata_raw = form.get("metadata")

    if not isinstance(image, UploadFile):
        raise HTTPException(status_code=400, detail="image is required")

    metadata = _parse_metadata(metadata_raw)
    payload = await image.read()

    if getattr(request.app.state, "redis", None) is None:
        raise HTTPException(status_code=503, detail="queue unavailable; ensure redis and smartbite-worker are running")

    service = _scan_service(request, session)
    try:
        scan_id = await service.create_scan(
            image_bytes=payload,
            filename=image.filename or "upload.jpg",
            content_type=image.content_type or "application/octet-stream",
            qr_code=f"oneshot-{uuid4().hex}",
            user_id="oneshot",
            metadata=metadata,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    settings = get_settings()
    timeout = max(1.0, float(settings.one_shot_timeout_seconds))
    poll_interval = max(0.1, float(settings.one_shot_poll_interval_seconds))
    deadline = monotonic() + timeout
    session_factory = get_session_factory()

    while monotonic() < deadline:
        async with session_factory() as poll_session:
            poll_service = _scan_service(request, poll_session)
            status, result = await poll_service.get_scan_payload(scan_id)
            if status == ScanStatus.DONE and result is not None:
                return OneShotAnalyzeResponse(scan_id=scan_id, status=status, result=result)
            if status == ScanStatus.FAILED:
                raise HTTPException(status_code=500, detail="one-shot scan failed during processing")
        await asyncio.sleep(poll_interval)

    raise HTTPException(
        status_code=504,
        detail=f"one-shot timed out after {timeout:.1f}s; ensure smartbite-worker is running",
    )


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
