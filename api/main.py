# api/main.py
"""
FastAPI application entrypoint for the AI Agentic Customer Support Platform.

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs (Swagger UI):  http://localhost:8000/docs
Alternative docs (ReDoc):       http://localhost:8000/redoc

Structure:
    api/
      main.py           <- this file: app instance, lifespan, router wiring
      schemas.py        <- HTTP request/response Pydantic models
      dependencies.py   <- singleton service construction (DI via lru_cache)
      routers/
        chat.py         <- POST /api/v1/chat
        health.py       <- GET  /api/v1/health

This file intentionally contains no business logic — every route delegates
to the same `services/` layer used by `main_simulation.py`, so behaviour is
identical whether the platform is driven via CLI simulation or this API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import warm_up_services
from api.routers import chat, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up all services (LLM, RAG, agents) once at startup, not on first request."""
    logger.info("[API] Starting up AI Agentic Customer Support Platform...")
    warm_up_services()
    yield
    logger.info("[API] Shutting down.")


app = FastAPI(
    title="AI Agentic Customer Support Platform",
    description=(
        "Multi-agent customer support backend: PII masking → LLM-based "
        "intent routing → specialised agents (order tracking, product "
        "recommendation, returns, general Q&A) with RAG-backed knowledge "
        "retrieval, and an escalation fallback."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: permissive by default for local development (Streamlit UI running on
# a different port). Tighten `allow_origins` to your actual frontend domain(s)
# before deploying to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(health.router)


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    """Basic liveness probe / landing endpoint."""
    return {
        "service": "AI Agentic Customer Support Platform",
        "status": "running",
        "docs": "/docs",
    }
