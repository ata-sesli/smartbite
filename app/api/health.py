from __future__ import annotations

from litestar import get

from app.domain.schemas import HealthResponse
from app.infra.clock import utcnow
from app.infra.settings import get_settings


@get('/health')
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status='ok', app=settings.app_name, timestamp=utcnow())
