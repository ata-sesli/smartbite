from __future__ import annotations

import uvicorn

from app.application import create_app

app = create_app()


def run() -> None:
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=False)
