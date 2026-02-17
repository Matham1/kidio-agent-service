"""
LLM Service — async client for the Ollama inference runtime.

Responsibilities:
    • Build the full prompt (system + RAG context + user message).
    • Call Ollama ``/api/generate`` with retries and timeout.
    • Parse and return the generated text.
    • Expose ``generate_structured`` for typed / JSON output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
import orjson
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


# ── Data transfer objects ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Encapsulates a single LLM generation result."""

    text: str
    model: str
    total_duration_ns: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0
    raw: dict[str, Any] | None = None


# ── Service ──────────────────────────────────────────────────────────


class LLMService:
    """Async LLM client that communicates with Ollama over HTTP."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle ----------------------------------------------------

    async def startup(self) -> None:
        """Create the shared ``httpx.AsyncClient``."""
        self._client = httpx.AsyncClient(
            base_url=self._settings.ollama_base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=self._settings.llm_timeout_seconds,
                write=10.0,
                pool=10.0,
            ),
        )
        logger.info(
            "llm_service.started",
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_model,
        )

    async def shutdown(self) -> None:
        """Close the HTTP connection pool."""
        if self._client:
            await self._client.aclose()
            logger.info("llm_service.shutdown")

    # -- prompt construction ------------------------------------------

    @staticmethod
    def build_prompt(
        *,
        system_prompt: str,
        context_json: str = "",
        rag_context: str = "",
        user_message: str,
    ) -> str:
        """
        Assemble the full prompt sent to the model.

        Injection order:
            1. system_prompt
            2. context_json  (structured business context from Django)
            3. rag_context   (retrieved document chunks)
            4. user_message
        """
        parts: list[str] = []

        if system_prompt:
            parts.append(f"[System]\n{system_prompt}")

        if context_json:
            parts.append(f"[Context]\n{context_json}")

        if rag_context:
            parts.append(rag_context)

        parts.append(f"[User]\n{user_message}")

        return "\n\n".join(parts)

    # -- generation ---------------------------------------------------

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """
        Send a generation request to Ollama and return the result.

        Retries on transient HTTP / connection errors with exponential backoff.
        """
        if self._client is None:
            raise RuntimeError("LLMService not started — call startup() first")

        _model = model or self._settings.ollama_model
        _temperature = (
            temperature if temperature is not None else self._settings.default_temperature
        )
        _max_tokens = max_tokens or self._settings.default_max_tokens

        payload: dict[str, Any] = {
            "model": _model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": _temperature,
                "num_predict": _max_tokens,
            },
        }

        logger.debug(
            "llm_service.request",
            model=_model,
            temperature=_temperature,
            max_tokens=_max_tokens,
            prompt_len=len(prompt),
        )

        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()

        data: dict[str, Any] = orjson.loads(response.content)

        result = LLMResult(
            text=data.get("response", ""),
            model=data.get("model", _model),
            total_duration_ns=data.get("total_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            raw=data,
        )

        logger.info(
            "llm_service.response",
            model=result.model,
            output_len=len(result.text),
            eval_count=result.eval_count,
        )
        return result

    # -- structured output helper -------------------------------------

    async def generate_structured(
        self,
        *,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate and attempt to parse the output as JSON.

        Falls back to ``{"text": <raw_output>}`` on parse failure.
        """
        result = await self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            parsed: dict[str, Any] = json.loads(result.text)
            return parsed
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "llm_service.json_parse_failed",
                output_preview=result.text[:200],
            )
            return {"text": result.text}
