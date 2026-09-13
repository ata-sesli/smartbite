from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment(tmp_path_factory: pytest.TempPathFactory) -> None:
    root = tmp_path_factory.mktemp("smartbite-test")
    db_path = root / "test.db"
    storage_root = root / "storage"

    import os

    os.environ["SMARTBITE_DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    os.environ["SMARTBITE_STORAGE_ROOT"] = str(storage_root)
    os.environ["SMARTBITE_REDIS_URL"] = "redis://localhost:6399/0"
    os.environ["SMARTBITE_OCR_DEVICE_MODE"] = "cpu"
    os.environ["SMARTBITE_MODEL_STARTUP_STRICT_VALIDATION"] = "false"
    os.environ["SMARTBITE_MODELS_AUTO_DOWNLOAD"] = "false"

    from app.infra import db as db_module
    from app.infra.settings import get_settings

    get_settings.cache_clear()
    db_module._engine = None
    db_module._session_factory = None

    from app.domain.models import Base
    from app.infra.db import get_engine

    engine: AsyncEngine = get_engine()

    async def setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(setup())

    yield

    async def teardown() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    asyncio.run(teardown())
