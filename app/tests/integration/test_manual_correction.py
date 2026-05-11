from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.domain.enums import ExpiryClassification
from app.domain.repositories import DomainRepository
from app.domain.services import ScanService
from app.infra.db import get_session_factory
from app.infra.settings import get_settings
from app.infra.storage import LocalStorage


@pytest.mark.asyncio
async def test_manual_correction_updates_classification() -> None:
    settings = get_settings()
    session_factory = get_session_factory()
    storage = LocalStorage(settings.storage_root)

    async with session_factory() as session:
        repo = DomainRepository(session)
        product = await repo.get_or_create_product("qr-manual")
        scan = await repo.create_scan(
            product_id=product.id,
            user_id="user-manual",
            image_path="raw/manual.jpg",
            metadata_json=None,
        )
        await repo.upsert_expiry_status(
            scan_id=scan.id,
            status=ExpiryClassification.MANUAL_REVIEW_REQUIRED,
            days_remaining=None,
            alert_required=True,
            needs_review=True,
            reason="initial",
        )
        await session.commit()

    async with session_factory() as session:
        service = ScanService(
            session,
            storage,
            redis=None,
            max_upload_size_bytes=settings.max_upload_size_bytes,
            accepted_file_types=settings.accepted_file_types,
            accepted_file_extensions=settings.accepted_file_extensions,
            alert_threshold_days=settings.alert_threshold_days,
            scan_job_expires_seconds=settings.scan_job_expires_seconds,
        )
        _, _, classification, days_remaining = await service.manual_correct_scan(
            scan.id,
            parsed_date=date.today() + timedelta(days=10),
            reason="operator fixed",
        )

    assert classification == ExpiryClassification.SAFE
    assert days_remaining is not None
