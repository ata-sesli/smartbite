from __future__ import annotations

import uvicorn

from app.application import create_app
from app.infra.settings import get_settings

app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.api_port, reload=False)
