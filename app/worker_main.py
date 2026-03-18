from __future__ import annotations

from arq.worker import Worker
from arq.connections import RedisSettings

from app.infra.settings import get_settings
from app.workers.scan_jobs import process_scan, startup


def run() -> None:
    settings = get_settings()
    worker = Worker(
        functions=[process_scan],
        on_startup=startup,
        redis_settings=RedisSettings.from_dsn(settings.redis_url),
        job_timeout=300,
        max_tries=settings.scan_retry_count,
    )
    worker.run()
