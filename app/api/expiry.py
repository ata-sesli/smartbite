from __future__ import annotations

from datetime import date
from uuid import UUID

from litestar import Request, get
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import ExpiryClassification
from app.domain.schemas import ExpiryItemResponse, ExpiryListResponse
from app.domain.services import ScanService
from app.infra.settings import get_settings
from app.infra.storage import LocalStorage


@get("/expiry")
async def list_expiry(
    request: Request,
    session: AsyncSession,
    status: ExpiryClassification | None = None,
    user_id: str | None = None,
    product_id: UUID | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> ExpiryListResponse:
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

    items = await service.list_expiry(
        status=status,
        user_id=user_id,
        product_id=product_id,
        date_from=date_from,
        date_to=date_to,
    )

    return ExpiryListResponse(
        items=[
            ExpiryItemResponse(
                scan_id=item.scan_id,
                user_id=item.user_id,
                product_id=item.product_id,
                parsed_date=item.parsed_date,
                expiry_classification=item.expiry_classification,
                days_remaining=item.days_remaining,
                needs_review=item.needs_review,
            )
            for item in items
        ]
    )
