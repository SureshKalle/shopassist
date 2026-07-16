# api/routers/health.py
"""
Health/status endpoint - used by Docker's healthcheck (see
docker-compose.yml) and for manual liveness checks.

GET /api/v1/health
"""

import logging
import os
import urllib.error
import urllib.request

from fastapi import APIRouter, Depends
from sqlalchemy import text

from api.config import settings
from api.dependencies import get_agents, get_ecommerce_client, get_rag_service
from api.schemas import HealthResponse
from clients.ecommerce_api_client import MockECommerceAPIClient
from services.rag import MockRAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/health", tags=["health"])


def _ollama_reachable(timeout: float = 3.0) -> bool:
    # OLLAMA_API_BASE_URL is a services/-owned var (services/llm_inference.py
    # reads it directly), not one of api/config.py's - read it the same way here.
    base_url = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout)
        return True
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("[health] Ollama not reachable: %s", exc)
        return False


@router.get("", response_model=HealthResponse)
def health_check(
    agents: dict = Depends(get_agents),
    rag_service: MockRAGService = Depends(get_rag_service),
    ecommerce_client: MockECommerceAPIClient = Depends(get_ecommerce_client),
) -> HealthResponse:
    """
    Never raises on a downstream outage — that should show up as
    database_reachable=False / llm_reachable=False, not a 500 that takes
    the health check down along with whatever it was trying to report on.
    """
    try:
        with ecommerce_client.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        database_reachable = True
    except Exception as exc:
        logger.warning("[health] Database not reachable: %s", exc)
        database_reachable = False

    return HealthResponse(
        status="ok",
        registered_agents=list(agents.keys()),
        rag_documents_indexed=len(rag_service.vector_db),
        database_reachable=database_reachable,
        llm_reachable=_ollama_reachable(),
        api_key_enforced=settings.api_key_enforce,
    )
