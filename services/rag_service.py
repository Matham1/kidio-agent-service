"""
RAG (Retrieval-Augmented Generation) service layer.

Defines the retriever abstraction and a dummy implementation.
Swap ``DummyRetriever`` for a real implementation backed by
pgvector / Qdrant / Weaviate when ready.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Sequence

import structlog

logger = structlog.get_logger(__name__)


# ── Data structures ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single retrieved document chunk."""

    content: str
    source: str = ""
    score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


# ── Abstract retriever ───────────────────────────────────────────────


class BaseRetriever(abc.ABC):
    """
    Abstract retriever interface.

    All concrete retrievers (pgvector, Qdrant, etc.) must implement
    ``retrieve``.  The orchestrator depends ONLY on this abstraction.
    """

    @abc.abstractmethod
    async def retrieve(
        self, query: str, *, top_k: int = 5
    ) -> Sequence[RetrievedChunk]:
        """Return the top-k most relevant chunks for *query*."""
        ...


# ── Dummy retriever (placeholder) ───────────────────────────────────


class DummyRetriever(BaseRetriever):
    """
    No-op retriever that always returns an empty list.

    Use this until a real vector-store retriever is wired in.

    **Where to plug in a real retriever:**

    1. Create ``services/pgvector_retriever.py`` (or ``qdrant_retriever.py``).
    2. Implement ``BaseRetriever.retrieve`` using your vector DB client.
    3. In ``services/orchestrator.py``, inject the new retriever instead
       of ``DummyRetriever``.  Example::

           from services.pgvector_retriever import PgVectorRetriever
           retriever = PgVectorRetriever(dsn=settings.pgvector_dsn)
           orchestrator = AgentOrchestrator(llm_service, retriever)
    """

    async def retrieve(
        self, query: str, *, top_k: int = 5
    ) -> Sequence[RetrievedChunk]:
        logger.debug("rag.dummy_retrieve", query=query[:80], top_k=top_k)
        return []


def format_rag_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks into a prompt-injectable context block."""
    if not chunks:
        return ""

    parts: list[str] = ["### Retrieved Context ###"]
    for idx, chunk in enumerate(chunks, 1):
        header = f"[{idx}] (source={chunk.source}, score={chunk.score:.3f})"
        parts.append(f"{header}\n{chunk.content}")
    parts.append("### End Context ###")
    return "\n\n".join(parts)
