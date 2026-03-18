from __future__ import annotations

from litestar import post
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas import AlertsProcessResponse
from app.domain.services import AlertService
from app.workers.alert_jobs import process_alert_delivery


@post("/alerts/process")
async def process_alerts(session: AsyncSession) -> AlertsProcessResponse:
    service = AlertService(session)
    created, skipped = await service.process_alerts()
    await process_alert_delivery()
    return AlertsProcessResponse(created=created, skipped_existing=skipped)
