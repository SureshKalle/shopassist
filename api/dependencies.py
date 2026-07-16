# api/dependencies.py
"""
Singleton service wiring for the FastAPI layer.

Each service/agent is built once per process via `@lru_cache`; FastAPI's
`Depends()` then resolves the same cached instance on every request. Mirrors
how main_simulation.py wires up the same services for the CLI simulation.
"""

import logging
from functools import lru_cache

from clients.ecommerce_api_client import MockECommerceAPIClient
from common.models import RawCustomerConversation, RawProductRecord
from services.agents.base_agent import BaseAgent
from services.agents.escalation_agent import EscalationAgent
from services.agents.general_purpose_agent import GeneralPurposeAgent
from services.agents.order_tracking_agent import OrderTrackingAgent
from services.agents.product_recommendation_agent import ProductRecommendationAgent
from services.data_pipeline import DataIngestionPipeline
from services.llm_inference import LLMInferenceService
from services.orchestrator import AgentOrchestratorService
from services.pii_masker import PIIMasker
from services.rag import MockRAGService

logger = logging.getLogger(__name__)

# Same sample data main_simulation.py ingests before its demo queries, so
# the RAG store isn't empty on this API's first request. No policy docs
# seeded - DataIngestionPipeline has no ingest_policy_documents method yet.
_SAMPLE_CONVERSATIONS = [
    RawCustomerConversation(
        id="conv_001",
        text="Hi, my name is John Doe, and I want to know about my order 12345.",
        metadata={"source": "twitter", "user_id": "jd_123"},
    ),
    RawCustomerConversation(
        id="conv_002",
        text="Can you help me with a return for product X? My email is john.doe@example.com.",
        metadata={"source": "web_form", "user_id": "jd_123"},
    ),
    RawCustomerConversation(
        id="conv_003",
        text="I love my new laptop! Is there a warranty?",
        metadata={"source": "web_chat", "user_id": "cust_002"},
    ),
]

_SAMPLE_PRODUCTS = [
    RawProductRecord(
        product_id="PROD_LAP_001",
        raw_description=(
            "High-performance gaming laptop with an i7 processor, 16GB RAM, "
            "and a 1TB SSD. Stunning display and RGB keyboard."
        ),
        specs={"CPU": "i7", "RAM": "16GB", "Storage": "1TB SSD"},
        reviews=["Great product!", "Fast delivery.", "Screen is amazing!"],
        price="$1200.00",
    ),
    RawProductRecord(
        product_id="PROD_HEAD_002",
        raw_description=(
            "Premium noise-cancelling headphones for immersive audio. "
            "Comfortable earcups and 20-hour battery life."
        ),
        specs={"Color": "Black", "Battery": "20h"},
        reviews=["Awesome sound!", "John Doe found them comfy and fit perfectly."],
        price="$250.00",
    ),
]


@lru_cache
def get_pii_masker() -> PIIMasker:
    return PIIMasker()


@lru_cache
def get_llm_service() -> LLMInferenceService:
    return LLMInferenceService()


@lru_cache
def get_rag_service() -> MockRAGService:
    return MockRAGService(get_llm_service())


@lru_cache
def get_ecommerce_client() -> MockECommerceAPIClient:
    return MockECommerceAPIClient()


@lru_cache
def get_data_pipeline() -> DataIngestionPipeline:
    return DataIngestionPipeline(get_pii_masker(), get_llm_service(), get_rag_service())

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
    return AgentOrchestratorService(get_llm_service(), get_pii_masker(), get_agents())


def warm_up_services() -> None:
    """Eagerly construct every singleton and seed the RAG store at startup
    (called from main.py's lifespan handler), instead of lazily on first
    request."""
    logger.info("[API] Warming up services...")
    get_orchestrator()  # transitively constructs everything else
    pipeline = get_data_pipeline()
    pipeline.ingest_customer_conversations(_SAMPLE_CONVERSATIONS)
    pipeline.ingest_product_catalog(_SAMPLE_PRODUCTS)
    logger.info("[API] Service warm-up complete.")
