## ✅ AI Agent Microservice

### Project Structure

```
ai_microservice/
├── core/
│   ├── __init__.py
│   ├── config.py          ← Pydantic v2 Settings (all env vars)
│   └── tracking.py        ← MLflow async context manager
├── proto/
│   ├── __init__.py
│   └── agent.proto        ← gRPC service + messages
├── services/
│   ├── __init__.py
│   ├── rag_service.py     ← BaseRetriever (abstract) + DummyRetriever
│   ├── llm_service.py     ← Async Ollama client (httpx + retries)
│   └── orchestrator.py    ← Central coordination: RAG → LLM → MLflow
├── transport/
│   ├── __init__.py
│   ├── grpc_server.py     ← gRPC AgentServiceServicer
│   └── rest_api.py        ← FastAPI router (POST /generate, GET /health)
├── main.py                ← FastAPI entrypoint (port 8002)
├── run_grpc.py            ← gRPC entrypoint (port 50051)
├── start.sh               ← Dual-server launcher (both ports)
├── requirements.txt
├── Dockerfile             ← python:3.11-slim, auto-compiles protobuf
├── docker-compose.yml     ← ai-agent + ollama + mlflow
├── .env.example
├── .env
├── .dockerignore
└── .gitignore
```

### Architecture

```
Django ──(gRPC :50051)──▶ ai-agent ──(HTTP)──▶ ollama (Qwen2.5-7B)
                              │
                              └──(HTTP)──▶ mlflow (:5000)
```

### Key Design Decisions

| Concern | Implementation |
|---|---|
| **Dual transport** | `start.sh` runs both FastAPI + gRPC concurrently in one container |
| **RAG abstraction** | `BaseRetriever` ABC → inject `PgVectorRetriever` / `QdrantRetriever` later |
| **LLM resilience** | `tenacity` retries (3x exponential backoff) + configurable timeout |
| **MLflow tracking** | Every generation logged (params, prompts, output, latency) via async context manager |
| **Structured output** | `generate_structured()` parses JSON, gracefully degrades |
| **GPU readiness** | Commented `deploy.resources.reservations` block in docker-compose |
| **Container-safe** | All URLs use container hostnames (`ollama:11434`, `mlflow:5000`) |

### 🚀 Execution Instructions

```bash
# 1. Build all images
docker compose build

# 2. Start the stack
docker compose up -d

# 3. Pull the LLM model into Ollama
docker exec -it ollama ollama pull qwen2.5:7b-instruct
```

### Service URLs

| Service | URL |
|---|---|
| REST API | `http://localhost:8002` |
| Health check | `http://localhost:8002/health` |
| gRPC | `localhost:50051` |
| MLflow UI | `http://localhost:5000` |
| Ollama | `http://localhost:11434` |

### Test the REST endpoint

```bash
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Explain quantum computing in 3 sentences.",
    "system_prompt": "You are a helpful science tutor.",
    "context_json": "{}",
    "agent_settings": {
      "model_name": "",
      "temperature": 0.7,
      "max_tokens": 512
    }
  }'
```

### Running Servers Separately

```bash
# REST only (default Dockerfile CMD)
docker compose up ai-agent  # uses: uvicorn main:app

# gRPC only (override CMD)
# In docker-compose.yml change command to: ["python", "run_grpc.py"]
```

### Where to Plug in a Real RAG Retriever

1. Create `services/pgvector_retriever.py` implementing `BaseRetriever.retrieve()`
2. Add your vector DB connection settings to `core/config.py`
3. In `services/orchestrator.py`, swap `DummyRetriever()` for your implementation
4. Set `RAG_ENABLED=true` in `.env`