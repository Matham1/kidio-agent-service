"""
FastAPI REST transport layer.

Provides:
    • ``POST /generate`` — mirrors the gRPC ``Generate`` RPC.
    • ``GET  /health``   — lightweight liveness / readiness probe.

Delegates all business logic to ``AgentOrchestrator``.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.config import get_settings

logger = structlog.get_logger(__name__)

router = APIRouter()

# ── Pydantic request / response models (mirror proto schema) ─────────


class AgentSettingsModel(BaseModel):
    """Optional tuning knobs sent by the caller."""

    model_name: str = Field(default="", description="Override the default LLM model")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=0, ge=0, le=32768)


class GenerateRequest(BaseModel):
    """Payload accepted by ``POST /generate``."""

    user_message: str = Field(..., min_length=1)
    system_prompt: str = Field(default="")
    context_json: str = Field(default="")
    agent_settings: AgentSettingsModel = Field(default_factory=AgentSettingsModel)


class GenerateResponse(BaseModel):
    """Returned by ``POST /generate``."""

    text: str
    metadata_json: str = Field(default="{}")


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = ""
    ollama_url: str = ""


# ── Endpoints ────────────────────────────────────────────────────────

# The orchestrator is injected at startup from ``main.py`` via
# ``set_orchestrator()``.  This avoids circular imports and keeps the
# transport layer decoupled.

_orchestrator: Any = None


def set_orchestrator(orchestrator: Any) -> None:
    """Called once from ``main.py`` during app lifespan."""
    global _orchestrator  # noqa: PLW0603
    _orchestrator = orchestrator


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness / readiness probe."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        ollama_url=settings.ollama_base_url,
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    """
    Generate text via the AI agent.

    Mirrors the gRPC ``Generate`` RPC so Django can call either transport
    interchangeably.
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")

    logger.info(
        "rest.generate.request",
        user_message_len=len(body.user_message),
    )

    try:
        result = await _orchestrator.generate(
            user_message=body.user_message,
            system_prompt=body.system_prompt,
            context_json=body.context_json,
            model_name=body.agent_settings.model_name or None,
            temperature=body.agent_settings.temperature or None,
            max_tokens=body.agent_settings.max_tokens or None,
        )
        return GenerateResponse(
            text=result["text"],
            metadata_json=result["metadata_json"],
        )
    except Exception as exc:
        logger.exception("rest.generate.error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
