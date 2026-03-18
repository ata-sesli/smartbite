from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging

from litestar import Litestar
from litestar.di import Provide
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import ROUTES
from app.infra.db import get_db_session
from app.infra.logging import configure_logging
from app.infra.queue import create_redis_pool
from app.infra.settings import get_settings
from app.infra.storage import LocalStorage

logger = logging.getLogger(__name__)


async def provide_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db_session():
        yield session


@asynccontextmanager
async def lifespan(app: Litestar):
    configure_logging()
    settings = get_settings()
    LocalStorage(settings.storage_root)
    try:
        app.state.redis = await create_redis_pool()
    except Exception as exc:  # pragma: no cover - runtime infra path
        logger.warning("redis unavailable during startup: %s", exc)
        app.state.redis = None
    try:
        yield
    finally:
        redis = getattr(app.state, 'redis', None)
        if redis is not None:
            await redis.close(close_connection_pool=True)


def create_app() -> Litestar:
    return Litestar(
        route_handlers=ROUTES,
        dependencies={'session': Provide(provide_session)},
        lifespan=[lifespan],
    )
