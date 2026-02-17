"""
Centralized application configuration.

All settings are loaded from environment variables via pydantic-settings.
Container-safe defaults — never references localhost.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service identity ──────────────────────────────────────────────
    service_name: str = Field(default="ai-agent", description="Logical service name")
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=False)

    # ── Ollama (LLM runtime) ─────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://ollama:11434",
        description="Base URL of the Ollama container",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b-instruct",
        description="Default LLM model name served by Ollama",
    )

    # ── LLM defaults ─────────────────────────────────────────────────
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=2048, ge=1, le=32768)
    llm_timeout_seconds: float = Field(
        default=120.0,
        description="HTTP timeout for a single Ollama call",
    )
    llm_max_retries: int = Field(
        default=3,
        description="Max retry attempts for transient Ollama failures",
    )

    # ── MLflow (observability) ────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        default="http://mlflow:5000",
        description="MLflow tracking server URI",
    )
    mlflow_experiment_name: str = Field(default="ai-agent-generations")

    # ── Transport ─────────────────────────────────────────────────────
    rest_host: str = Field(default="0.0.0.0")
    rest_port: int = Field(default=8002)
    grpc_host: str = Field(default="0.0.0.0")
    grpc_port: int = Field(default=50051)
    grpc_max_workers: int = Field(default=10)

    # ── RAG (placeholder — swap with pgvector / Qdrant config later) ─
    rag_enabled: bool = Field(default=False)
    rag_top_k: int = Field(default=5, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor — import and call ``get_settings()`` anywhere."""
    return Settings()
