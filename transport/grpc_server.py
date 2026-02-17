"""
gRPC transport layer — ``AgentService`` implementation.

Delegates all business logic to ``AgentOrchestrator``.
No LLM / RAG concerns live here.
"""

from __future__ import annotations

import asyncio
from concurrent import futures
from typing import TYPE_CHECKING

import grpc
import structlog
from grpc import aio as grpc_aio

# These modules are generated at Docker build time by grpc_tools.protoc.
# They will NOT exist on disk until ``protoc`` runs.
import proto.agent_pb2 as agent_pb2  # type: ignore[import-untyped]
import proto.agent_pb2_grpc as agent_pb2_grpc  # type: ignore[import-untyped]

from core.config import get_settings
from services.llm_service import LLMService
from services.orchestrator import AgentOrchestrator

if TYPE_CHECKING:
    from grpc import aio as _aio

logger = structlog.get_logger(__name__)


class AgentServicer(agent_pb2_grpc.AgentServiceServicer):
    """gRPC servicer that wraps the orchestrator."""

    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        self._orchestrator = orchestrator

    async def Generate(
        self,
        request: agent_pb2.GenerateRequest,
        context: _aio.ServicerContext,
    ) -> agent_pb2.GenerateResponse:
        """Handle a ``Generate`` RPC call."""
        logger.info(
            "grpc.generate.request",
            user_message_len=len(request.user_message),
        )

        try:
            # Extract settings from the request (fall back to defaults)
            settings = request.agent_settings
            result = await self._orchestrator.generate(
                user_message=request.user_message,
                system_prompt=request.system_prompt,
                context_json=request.context_json,
                model_name=settings.model_name or None,
                temperature=settings.temperature or None,
                max_tokens=settings.max_tokens or None,
            )

            return agent_pb2.GenerateResponse(
                text=result["text"],
                metadata_json=result["metadata_json"],
            )

        except Exception as exc:
            logger.exception("grpc.generate.error")
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Generation failed: {exc}",
            )
            # The above ``abort`` raises, but mypy doesn't know that.
            raise  # pragma: no cover


async def serve_grpc() -> None:
    """Boot the async gRPC server."""
    settings = get_settings()

    # Build dependencies
    llm_service = LLMService(settings)
    orchestrator = AgentOrchestrator(llm_service, settings=settings)
    await orchestrator.startup()

    server = grpc_aio.server(
        futures.ThreadPoolExecutor(max_workers=settings.grpc_max_workers),
    )
    agent_pb2_grpc.add_AgentServiceServicer_to_server(
        AgentServicer(orchestrator), server
    )

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)
    await server.start()

    logger.info("grpc_server.started", address=listen_addr)

    try:
        await server.wait_for_termination()
    finally:
        await orchestrator.shutdown()
        logger.info("grpc_server.stopped")
