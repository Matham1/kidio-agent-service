# ─────────────────────────────────────────────────────────────────────
# AI Agent Microservice — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────
# Build:  docker compose build ai-agent
# Run:    docker compose up ai-agent
# ─────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────
COPY . .

# ── Compile protobuf ─────────────────────────────────────────────────
RUN python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./proto \
    --grpc_python_out=./proto \
    proto/agent.proto && \
    # Fix relative import in generated gRPC stub
    sed -i 's/^import agent_pb2/from proto import agent_pb2/' proto/agent_pb2_grpc.py

# ── Expose ports ─────────────────────────────────────────────────────
EXPOSE 8002 50051

# ── Default entrypoint: FastAPI ──────────────────────────────────────
# To run gRPC instead:  CMD ["python", "run_grpc.py"]
# To run BOTH, override with the start.sh script (see below).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
