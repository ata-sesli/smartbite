from __future__ import annotations

import pytest

from app.domain.enums import ExpiryClassification, ScanStatus
from app.domain.repositories import DomainRepository
from app.domain.services import AlertService
from app.infra.db import get_session_factory


@pytest.mark.asyncio
async def test_alert_processing_is_idempotent() -> None:
    session_factory = get_session_factory()

    async with session_factory() as session:
        repo = DomainRepository(session)
        product = await repo.get_or_create_product("qr-idem")
        scan = await repo.create_scan(product_id=product.id, user_id="user-a", image_path="raw/a.jpg", metadata_json=None)
        await repo.upsert_expiry_status(
            scan_id=scan.id,
            status=ExpiryClassification.EXPIRING_SOON,
            days_remaining=2,
            alert_required=True,
            needs_review=False,
            reason="test",
        )
        scan.status = ScanStatus.DONE
        await session.commit()

    async with session_factory() as session:
        service = AlertService(session)
        created_1, skipped_1 = await service.process_alerts()

    async with session_factory() as session:
        service = AlertService(session)
        created_2, skipped_2 = await service.process_alerts()

    assert created_1 == 1
    assert skipped_1 == 0
    assert created_2 == 0
    assert skipped_2 >= 1
