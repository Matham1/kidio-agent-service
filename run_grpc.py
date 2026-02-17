"""
Standalone gRPC server entrypoint.

Run with::

    python run_grpc.py

Or start both servers concurrently::

    # Terminal 1 — REST
    uvicorn main:app --host 0.0.0.0 --port 8002

    # Terminal 2 — gRPC
    python run_grpc.py

Inside Docker the recommended approach is a single container per server,
or a supervisor / shell script that launches both (see Dockerfile CMD
and docker-compose overrides).
"""

from __future__ import annotations

import asyncio
import sys

import structlog

logger = structlog.get_logger(__name__)


def main() -> None:
    """Bootstrap and run the async gRPC server."""
    from transport.grpc_server import serve_grpc

    logger.info("run_grpc.starting")
    try:
        asyncio.run(serve_grpc())
    except KeyboardInterrupt:
        logger.info("run_grpc.interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
