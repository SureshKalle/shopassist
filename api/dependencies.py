# api/dependencies.py
"""
Singleton service wiring for the FastAPI layer.

All services/agents are expensive to construct (embedding models, LLM
clients, vectorstore connections, LangGraph agent compilation) so each is
built exactly once per process via `@lru_cache`, and FastAPI's `Depends()`
resolves the same cached instance on every request — no per-request
re-initialisation, no global mutable state scattered across route handlers.

This mirrors exactly how `main_simulation.py` wires the same services
together for the CLI simulation; the two entry points (API and simulation
script) share the same construction logic conceptually, just via different
mechanisms (FastAPI DI vs. plain function calls).
"""

import logging
import os
from functools import lru_cache

from services.ecommerce_client import DEFAULT_DB_URL, ECommerceAPIClient
from services.agents.base_agent import BaseAgent
from services.agents.escalation_agent import EscalationAgent
from services.agents.general_purpose_agent import GeneralPurposeAgent
from services.agents.order_tracking_agent import OrderTrackingAgent
from services.agents.product_recommendation_agent import ProductRecommendationAgent
from services.classifier_client import ClassifierClient
from services.data_pipeline import DataIngestionPipeline
from services.llm_inference import MockLLMInferenceService
from services.orchestrator import AgentOrchestratorService
from services.pii_masker import PIIMasker
from services.rag import BaseRAGService, MockRAGService, PgVectorRAGService

logger = logging.getLogger(__name__)


@lru_cache
def get_pii_masker() -> PIIMasker:
    return PIIMasker()


@lru_cache
def get_llm_service() -> MockLLMInferenceService:
    return MockLLMInferenceService()


@lru_cache
def get_rag_service() -> BaseRAGService:
    """PgVectorRAGService when DATABASE_URL is Postgres (real semantic
    search against shopassist-database's document_chunks table);
    MockRAGService otherwise (SQLite/unset — mirrors
    ECommerceAPIClient's own SQLite fallback exactly, so both services
    agree on which backend is active).
    """
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)
    if database_url.startswith("postgresql"):
        return PgVectorRAGService(get_llm_service(), database_url)
    return MockRAGService(get_llm_service())


@lru_cache
def get_ecommerce_client() -> ECommerceAPIClient:
    return ECommerceAPIClient()


@lru_cache
def get_classifier_client() -> ClassifierClient:
    return ClassifierClient()


@lru_cache
def get_data_pipeline() -> DataIngestionPipeline:
    return DataIngestionPipeline(get_pii_masker(), get_llm_service(), get_rag_service(), get_classifier_client())

@lru_cache
def get_agents() -> dict[str, BaseAgent]:
    deps = (get_llm_service(), get_rag_service(), get_ecommerce_client(), get_pii_masker())
    agents = {
        "OrderTrackingAgent": OrderTrackingAgent(*deps),
        "ProductRecommendationAgent": ProductRecommendationAgent(*deps),
        "GeneralPurposeAgent": GeneralPurposeAgent(*deps),
        "EscalationAgent": EscalationAgent(*deps),
    }
    logger.info("[API] Registered agents: %s", list(agents.keys()))
    return agents


@lru_cache
def get_orchestrator() -> AgentOrchestratorService:
    return AgentOrchestratorService(get_llm_service(), get_pii_masker(), get_agents(), get_ecommerce_client())


def warm_up_services() -> None:
    """
    Eagerly construct every singleton at app startup (called from the
    FastAPI lifespan handler in main.py) rather than lazily on first
    request — surfaces configuration errors (bad API key, unreachable
    Ollama server, etc.) at boot time instead of on a customer's first
    request, and avoids a slow "cold" first request while agent graphs
    compile and the embedding model loads.
    """
    logger.info("[API] Warming up services...")
    get_orchestrator()  # transitively constructs everything else
    get_data_pipeline()
    logger.info("[API] Service warm-up complete.")
