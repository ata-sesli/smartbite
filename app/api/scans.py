from __future__ import annotations

import asyncio
from io import BytesIO
import json
import mimetypes
from pathlib import Path
from time import monotonic
from uuid import UUID
from uuid import uuid4

from litestar import Request, get, patch, post, put
from litestar.datastructures import UploadFile
from litestar.exceptions import HTTPException
from litestar.response import Response
from PIL import Image, ImageOps
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import FinalResultStatus, ScanStatus
from app.domain.schemas import (
    ManualCorrectionRequest,
    ManualCorrectionResponse,
    OneShotAnalyzeResponse,
    QueueTestImagesResponse,
    ScanCreateResponse,
    ScanGetResponse,
    ScanListResponse,
    Test64ExpectedDateListResponse,
    Test64ExpectedDateUpsertRequest,
    Test64ExpectedDateUpsertResponse,
    Test64TruthBBoxManifestResponse,
    Test64TruthBBoxUpsertRequest,
    Test64TruthBBoxUpsertResponse,
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
        scan_job_expires_seconds=settings.scan_job_expires_seconds,
        general_text_ocr_timeout_seconds=settings.general_text_ocr_timeout_seconds,
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


def _parse_scan_status(value: str | None) -> ScanStatus | None:
    if not value:
        return None
    try:
        return ScanStatus(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"unsupported scan status: {value}") from exc


def _parse_final_status(value: str | None) -> FinalResultStatus | None:
    if not value:
        return None
    try:
        return FinalResultStatus(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"unsupported final status: {value}") from exc


def _media_type_for_path(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _oriented_image_response(path: str) -> Response[bytes]:
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image)
        output = BytesIO()
        suffix = Path(path).suffix.lower()
        if suffix == ".png":
            oriented.save(output, format="PNG")
            media_type = "image/png"
        elif suffix == ".webp":
            oriented.save(output, format="WEBP")
            media_type = "image/webp"
        else:
            oriented = oriented.convert("RGB")
            oriented.save(output, format="JPEG", quality=95)
            media_type = "image/jpeg"
    return Response(content=output.getvalue(), media_type=media_type)


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


@post("/test-images/queue")
async def queue_test_images_standalone(request: Request, session: AsyncSession) -> QueueTestImagesResponse:
    service = _scan_service(request, session)
    try:
        response = await service.queue_test_images()
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response


@get("/test64/labels")
async def list_test64_labels(request: Request, session: AsyncSession) -> Test64ExpectedDateListResponse:
    service = _scan_service(request, session)
    try:
        return await service.list_test64_expected_dates()
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@put("/test64/labels")
async def put_test64_label(
    request: Request,
    session: AsyncSession,
    data: Test64ExpectedDateUpsertRequest,
) -> Test64ExpectedDateUpsertResponse:
    filename = data.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")

    if data.month is None or not (1 <= data.month <= 12):
        raise HTTPException(status_code=400, detail="month must be between 1 and 12")
    if data.year is None or not (2000 <= data.year <= 2200):
        raise HTTPException(status_code=400, detail="year must be between 2000 and 2200")
    if data.day is not None and not (1 <= data.day <= 31):
        raise HTTPException(status_code=400, detail="day must be between 1 and 31 when provided")

    service = _scan_service(request, session)
    try:
        return await service.upsert_test64_expected_date(
            filename=filename,
            day=data.day,
            month=data.month,
            year=data.year,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@get("/test64/images/{filename:str}")
async def get_test64_image(request: Request, session: AsyncSession, filename: str) -> Response[bytes]:
    service = _scan_service(request, session)
    try:
        image_path = await service.get_test64_image_path(filename)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _oriented_image_response(image_path)


@get("/test64/truth-bboxes")
async def list_test64_truth_bboxes(request: Request, session: AsyncSession) -> Test64TruthBBoxManifestResponse:
    service = _scan_service(request, session)
    try:
        return await service.get_test64_truth_bboxes()
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@get("/test64/manual-crop-recognition-review")
async def get_manual_crop_recognition_review(
    request: Request,
    session: AsyncSession,
    include_success: bool = False,
) -> dict[str, object]:
    service = _scan_service(request, session)
    try:
        return await service.get_manual_crop_recognition_review(include_success=include_success)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@get("/test64/detection-review")
async def get_test64_detection_review(request: Request, session: AsyncSession) -> dict[str, object]:
    service = _scan_service(request, session)
    try:
        return await service.get_test64_detection_review()
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@get("/test64/full-pipeline-review")
async def get_test64_full_pipeline_review(request: Request, session: AsyncSession) -> dict[str, object]:
    service = _scan_service(request, session)
    try:
        return await service.get_test64_full_pipeline_review()
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@get("/test64/manual-crop-recognition-review/assets")
async def get_manual_crop_recognition_review_asset(
    request: Request,
    session: AsyncSession,
    run_id: str,
    path: str,
) -> Response[bytes]:
    service = _scan_service(request, session)
    try:
        asset_path = service.get_manual_crop_recognition_asset_path(run_id=run_id, relative_path=path)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=asset_path.read_bytes(), media_type=_media_type_for_path(str(asset_path)))


@put("/test64/truth-bboxes/{filename:str}")
async def put_test64_truth_bbox(
    request: Request,
    session: AsyncSession,
    filename: str,
    data: Test64TruthBBoxUpsertRequest,
) -> Test64TruthBBoxUpsertResponse:
    service = _scan_service(request, session)
    try:
        return await service.upsert_test64_truth_bbox(
            filename=filename,
            bbox_xyxy=data.true_bbox_xyxy,
            polygon_xy=data.true_polygon_xy,
            rotation_degrees=data.annotation_rotation_degrees,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@get("/scan-history")
async def list_scans(
    request: Request,
    session: AsyncSession,
    limit: int = 50,
    status: str | None = None,
    final_status: str | None = None,
    user_id: str | None = None,
) -> ScanListResponse:
    bounded_limit = min(max(limit, 1), 200)
    service = _scan_service(request, session)
    items = await service.list_scan_summaries(
        limit=bounded_limit,
        status=_parse_scan_status(status),
        final_status=_parse_final_status(final_status),
        user_id=user_id.strip() if user_id else None,
    )
    return ScanListResponse(items=items)


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


@get("/scans/{scan_id:uuid}/image")
async def get_scan_image(request: Request, session: AsyncSession, scan_id: UUID) -> Response[bytes]:
    service = _scan_service(request, session)
    try:
        image_path = await service.get_scan_image_path(scan_id)
        payload = Path(image_path).read_bytes() if Path(image_path).is_absolute() else service.storage.read_bytes(image_path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="image file not found") from exc
    return Response(content=payload, media_type=_media_type_for_path(image_path))


@get("/scans/{scan_id:uuid}/roi")
async def get_scan_roi(request: Request, session: AsyncSession, scan_id: UUID) -> Response[bytes]:
    service = _scan_service(request, session)
    try:
        roi_path = await service.get_scan_roi_path(scan_id)
        payload = service.storage.read_bytes(roi_path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "roi not available") from exc
    return Response(content=payload, media_type=_media_type_for_path(roi_path))


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
