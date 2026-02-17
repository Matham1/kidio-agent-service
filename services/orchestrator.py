"""
Agent Orchestrator — central coordination layer.

Wires together:
    • RAG retriever  (abstract — ``BaseRetriever``)
    • LLM service    (Ollama client)
    • MLflow tracking

No transport concerns live here.  gRPC and REST both call
``AgentOrchestrator.generate()``.
"""

from __future__ import annotations

from typing import Any

import orjson
import structlog

from core.config import Settings, get_settings
from core.tracking import track_generation
from services.llm_service import LLMService
from services.rag_service import (
    BaseRetriever,
    DummyRetriever,
    format_rag_context,
)

logger = structlog.get_logger(__name__)


class AgentOrchestrator:
    """
    Façade that receives a user request, enriches it via RAG,
    delegates to the LLM, and returns the final answer.
    """

    def __init__(
        self,
        llm_service: LLMService,
        retriever: BaseRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm_service
        self._retriever: BaseRetriever = retriever or DummyRetriever()
        self._settings = settings or get_settings()

    # -- lifecycle helpers (called from entrypoints) -------------------

    async def startup(self) -> None:
        await self._llm.startup()
        logger.info("orchestrator.started")

    async def shutdown(self) -> None:
        await self._llm.shutdown()
        logger.info("orchestrator.shutdown")

    # -- main entry point ---------------------------------------------

    async def generate(
        self,
        *,
        user_message: str,
        system_prompt: str = "",
        context_json: str = "",
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        End-to-end generation pipeline.

        1. Retrieve RAG context (if enabled / retriever is real).
        2. Build the full prompt.
        3. Call the LLM.
        4. Log everything to MLflow.
        5. Return ``{"text": ..., "metadata": {...}}``.
        """
        _model = model_name or self._settings.ollama_model
        _temperature = (
            temperature
            if temperature is not None
            else self._settings.default_temperature
        )
        _max_tokens = max_tokens or self._settings.default_max_tokens

        # 1 — RAG retrieval
        chunks = await self._retriever.retrieve(
            user_message, top_k=self._settings.rag_top_k
        )
        rag_context = format_rag_context(chunks)

        # 2 — Prompt assembly
        full_prompt = LLMService.build_prompt(
            system_prompt=system_prompt,
            context_json=context_json,
            rag_context=rag_context,
            user_message=user_message,
        )

        # 3 + 4 — LLM call wrapped in MLflow tracking
        async with track_generation(
            model_name=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
            user_message=user_message,
            system_prompt=system_prompt,
            full_prompt=full_prompt,
        ) as result_bag:
            llm_result = await self._llm.generate(
                prompt=full_prompt,
                model=_model,
                temperature=_temperature,
                max_tokens=_max_tokens,
            )
            result_bag["output"] = llm_result.text

        # 5 — Build response
        metadata: dict[str, Any] = {
            "model": llm_result.model,
            "temperature": _temperature,
            "max_tokens": _max_tokens,
            "eval_count": llm_result.eval_count,
            "prompt_eval_count": llm_result.prompt_eval_count,
            "total_duration_ns": llm_result.total_duration_ns,
            "rag_chunks_used": len(chunks),
        }

        return {
            "text": llm_result.text,
            "metadata": metadata,
            "metadata_json": orjson.dumps(metadata).decode(),
        }
