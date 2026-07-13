# api/routers/health.py
"""
Health/status endpoint — for load balancers, container orchestrators, and
uptime monitoring to check the service is actually ready before routing
real traffic to it.

GET /api/v1/health
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text

from api.dependencies import get_agents, get_ecommerce_client, get_rag_service
from api.schemas import HealthResponse
from services.ecommerce_client import ECommerceAPIClient
from services.rag import MockRAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health_check(
    agents: dict = Depends(get_agents),
    rag_service: MockRAGService = Depends(get_rag_service),
    ecommerce_client: ECommerceAPIClient = Depends(get_ecommerce_client),
) -> HealthResponse:
    """Report service status. Never raises on a downstream outage — that
    should show up as database_reachable=False, not a 500 that takes out
    the health check along with whatever it was trying to report on."""
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
        rag_collection_size=rag_service.collection_size(),
        database_reachable=database_reachable,
    )
