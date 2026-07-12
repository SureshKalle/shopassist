# api/routers/health.py
"""
Health/status endpoint — used by load balancers, Docker healthchecks, and
the Streamlit UI to display current backend configuration.

GET /api/v1/health
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_agents, get_rag_service
from api.schemas import HealthResponse
from common.config import EMBEDDING_PROVIDER, LLM_PROVIDER, VECTOR_STORE
from common.llm_factory import build_chat_model
from services.rag import RAGService

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health_check(
    rag_service: RAGService = Depends(get_rag_service),
    agents: dict = Depends(get_agents),
) -> HealthResponse:
    """Report service status and current configuration — no secrets exposed."""
    llm = build_chat_model()
    return HealthResponse(
        status="ok",
        llm_provider=LLM_PROVIDER,
        llm_available=llm is not None,
        embedding_provider=EMBEDDING_PROVIDER,
        vector_store=VECTOR_STORE,
        rag_collection_size=rag_service.collection_size(),
        registered_agents=list(agents.keys()),
    )
