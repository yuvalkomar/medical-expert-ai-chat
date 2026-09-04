from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api import router
from backend.app.config import Settings, get_settings
from backend.app.database import Database
from backend.app.llm.base import LLMProvider
from backend.app.llm.factory import create_provider
from backend.app.logging_config import configure_logging
from backend.app.streaming import StreamRegistry
from backend.app.workers import WorkerPool


def create_app(
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger = configure_logging(app_settings)
        database = Database(app_settings.database_url)
        database.create_schema()
        llm_provider = provider or create_provider(app_settings)
        stream_registry = StreamRegistry()
        worker_pool = WorkerPool(
            database=database,
            provider=llm_provider,
            settings=app_settings,
            logger=logger,
            stream_registry=stream_registry,
        )
        app.state.settings = app_settings
        app.state.database = database
        app.state.provider = llm_provider
        app.state.stream_registry = stream_registry
        app.state.worker_pool = worker_pool
        await worker_pool.start()
        try:
            yield
        finally:
            await worker_pool.stop()
            database.dispose()

    app = FastAPI(
        title="Medical Expert AI Chat",
        version="1.0.0",
        description="Asynchronous medical information chat service",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
