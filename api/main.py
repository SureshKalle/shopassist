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
        chat.py         <- POST/GET/DELETE /api/v1/chat...

This file intentionally contains no business logic — every route delegates
to the same `services/` layer used by `main_simulation.py`, so behaviour is
identical whether the platform is driven via CLI simulation or this API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.dependencies import warm_up_services
from api.middleware import RequestLoggingMiddleware
from api.routers import chat, health

# See .env.example / api/config.py for all api-owned settings.
# LOG_LEVEL=DEBUG also enables full request/response body logging (api/middleware.py).
logging.basicConfig(
    level=settings.log_level,
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
    title=settings.app_title,
    description=(
        "Multi-agent customer support backend: PII masking → LLM-based "
        "intent routing → specialised agents (order tracking, product "
        "recommendation, returns, general Q&A) with RAG-backed knowledge "
        "retrieval, and an escalation fallback."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

# Defaults to the Streamlit client's dev origin (:8501); override via
# CORS_ALLOWED_ORIGINS. Must be explicit origins, not "*" - invalid
# together with allow_credentials=True per the CORS spec.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Added after CORSMiddleware so it's outermost (Starlette layers middleware
# in reverse of add-order) and logs the request/response exactly as the
# client sees them, CORS headers included.
app.add_middleware(RequestLoggingMiddleware)

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
