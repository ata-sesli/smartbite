from __future__ import annotations

from litestar import get
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.repositories import DomainRepository
from app.domain.schemas import AdminMetricsResponse


@get('/admin/metrics')
async def metrics(session: AsyncSession) -> AdminMetricsResponse:
    repo = DomainRepository(session)
    payload = await repo.metrics()
    return AdminMetricsResponse(**payload)
