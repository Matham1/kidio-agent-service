"""
FastAPI application entrypoint.

Boots the FastAPI server with a managed lifespan that initialises and
tears down the orchestrator (and therefore the LLM client).

Run with::

    uvicorn main:app --host 0.0.0.0 --port 8002

Or use the Dockerfile ``CMD`` which does exactly this.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
import uvicorn
from fastapi import FastAPI

from core.config import get_settings
from services.llm_service import LLMService
from services.orchestrator import AgentOrchestrator
from transport.rest_api import router, set_orchestrator

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage orchestrator lifecycle alongside the FastAPI app."""
    settings = get_settings()

    llm_service = LLMService(settings)
    orchestrator = AgentOrchestrator(llm_service, settings=settings)
    await orchestrator.startup()
    set_orchestrator(orchestrator)

    logger.info(
        "fastapi.lifespan.started",
        service=settings.service_name,
        rest_port=settings.rest_port,
    )

    yield  # app is running

    await orchestrator.shutdown()
    logger.info("fastapi.lifespan.shutdown")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title=f"{settings.service_name} API",
        version="1.0.0",
        description="AI Agent microservice — REST interface",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    _settings = get_settings()
    uvicorn.run(
        "main:app",
        host=_settings.rest_host,
        port=_settings.rest_port,
        log_level="info",
    )
