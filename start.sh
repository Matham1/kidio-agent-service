#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# start.sh — Launch both FastAPI (REST) and gRPC servers concurrently
# Used when a single container must expose both transports.
#
# Usage in docker-compose override:
#   command: ["bash", "start.sh"]
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "▶ Starting FastAPI server on :8002 ..."
uvicorn main:app --host 0.0.0.0 --port 8002 &
FASTAPI_PID=$!

echo "▶ Starting gRPC server on :50051 ..."
python run_grpc.py &
GRPC_PID=$!

# Wait for either process to exit, then kill the other
trap "kill $FASTAPI_PID $GRPC_PID 2>/dev/null" EXIT
wait -n
exit $?
