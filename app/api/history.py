from __future__ import annotations

from typing import Any

from litestar import Request, get
from litestar.exceptions import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import FinalResultStatus, ScanStatus
from app.domain.services import ScanService
from app.infra.settings import get_settings
from app.infra.storage import LocalStorage


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


@get("/history/scans")
async def list_scan_history(
    request: Request,
    session: AsyncSession,
    limit: int = 50,
    status: str | None = None,
    final_status: str | None = None,
    user_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    bounded_limit = min(max(limit, 1), 200)
    settings = get_settings()
    service = ScanService(
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
    items = await service.list_scan_summaries(
        limit=bounded_limit,
        status=_parse_scan_status(status),
        final_status=_parse_final_status(final_status),
        user_id=user_id.strip() if user_id else None,
    )
    return {"items": [item.model_dump(mode="json") for item in items]}
