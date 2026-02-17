"""
MLflow tracking helpers.

Provides a context-manager that wraps every LLM generation call,
logging parameters, prompts, outputs, and latency to the configured
MLflow tracking server.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import mlflow
import structlog

from core.config import get_settings

logger = structlog.get_logger(__name__)

_EXPERIMENT_ENSURED = False


def _ensure_experiment() -> None:
    """Create / set the MLflow experiment once per process."""
    global _EXPERIMENT_ENSURED  # noqa: PLW0603
    if _EXPERIMENT_ENSURED:
        return

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    _EXPERIMENT_ENSURED = True
    logger.info(
        "mlflow.experiment_configured",
        tracking_uri=settings.mlflow_tracking_uri,
        experiment=settings.mlflow_experiment_name,
    )


@asynccontextmanager
async def track_generation(
    *,
    model_name: str,
    temperature: float,
    max_tokens: int,
    user_message: str,
    system_prompt: str,
    full_prompt: str,
) -> AsyncIterator[dict[str, Any]]:
    """
    Async context manager for tracking an LLM generation in MLflow.

    Usage::

        async with track_generation(...) as result:
            result["output"] = await llm.generate(...)

    On exit the context manager logs everything to MLflow.
    ``result`` is a mutable dict — set ``result["output"]`` inside the block.
    """
    _ensure_experiment()

    result: dict[str, Any] = {"output": "", "error": None}
    start = time.perf_counter()

    try:
        yield result
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        latency = time.perf_counter() - start
        try:
            with mlflow.start_run(nested=True):
                mlflow.log_params(
                    {
                        "model_name": model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                )
                mlflow.log_text(user_message, "user_message.txt")
                mlflow.log_text(system_prompt, "system_prompt.txt")
                mlflow.log_text(full_prompt, "full_prompt.txt")
                mlflow.log_text(str(result["output"]), "generation_output.txt")
                mlflow.log_metrics(
                    {
                        "latency_seconds": latency,
                        "output_length": len(str(result["output"])),
                    }
                )
                if result["error"]:
                    mlflow.log_text(result["error"], "error.txt")
                    mlflow.set_tag("status", "error")
                else:
                    mlflow.set_tag("status", "success")
        except Exception:
            logger.exception("mlflow.logging_failed")
