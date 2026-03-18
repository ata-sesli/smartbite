from __future__ import annotations

import logging

import httpx

from app.domain.repositories import DomainRepository
from app.infra.db import get_session_factory
from app.infra.settings import get_settings

logger = logging.getLogger(__name__)


async def process_alert_delivery() -> dict[str, int]:
    settings = get_settings()
    sent = 0
    failed = 0

    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = DomainRepository(session)
        pending = await repo.list_pending_alert_events()

        async with httpx.AsyncClient(timeout=5.0) as client:
            for event in pending:
                try:
                    await client.post(settings.webhook_stub_url, json=event.payload_json)
                    await repo.mark_alert_delivery(event.id, success=True)
                    sent += 1
                except Exception:  # pragma: no cover - network path
                    await repo.mark_alert_delivery(event.id, success=False)
                    failed += 1

        await session.commit()

    return {"sent": sent, "failed": failed}
