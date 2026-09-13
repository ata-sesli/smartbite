from __future__ import annotations

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

from app.infra.settings import get_settings

SCAN_JOB_NAME = "process_scan"
ALERT_JOB_NAME = "process_alerts"


async def create_redis_pool() -> ArqRedis:
    settings = get_settings()
    redis_url = settings.redis_url

    redis_settings = RedisSettings.from_dsn(redis_url)
    return await create_pool(redis_settings)
