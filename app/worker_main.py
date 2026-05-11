from __future__ import annotations

import os

# Ensure Paddle does not perform remote model-hoster reachability checks on startup.
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from arq.worker import Worker
from arq.connections import RedisSettings

from app.infra.settings import get_settings
from app.workers.scan_jobs import process_general_text_ocr, process_scan, startup


def run() -> None:
    settings = get_settings()
    worker = Worker(
        functions=[process_scan, process_general_text_ocr],
        on_startup=startup,
        redis_settings=RedisSettings.from_dsn(settings.redis_url),
        job_timeout=300,
        max_tries=settings.scan_retry_count,
        max_jobs=settings.worker_max_jobs,
    )
    worker.run()
