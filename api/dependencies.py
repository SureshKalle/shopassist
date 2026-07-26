# api/dependencies.py
"""
Singleton service wiring for the FastAPI layer.

Each service/agent is built once per process via `@lru_cache`; FastAPI's
`Depends()` then resolves the same cached instance on every request. Mirrors
how main_simulation.py wires up the same services for the CLI simulation.
"""

import logging
from functools import lru_cache

from clients.ecommerce_api_client import EcommerceClient
from common.models import RawCustomerConversation, RawProductRecord
from services.agents.base_agent import BaseAgent
from services.agents.escalation_agent import EscalationAgent
from services.agents.general_purpose_agent import GeneralPurposeAgent
from services.agents.order_tracking_agent import OrderTrackingAgent
from services.agents.product_recommendation_agent import ProductRecommendationAgent
from services.classifier_client import ClassifierClient
from services.data_pipeline import DataIngestionPipeline
from services.llm_inference import LLMInferenceService
from services.orchestrator import AgentOrchestratorService
from services.pii_masker import PIIMasker
from services.rag import RAGService

logger = logging.getLogger(__name__)

# Same sample data main_simulation.py ingests before its demo queries, so
# the RAG store isn't empty on this API's first request. Policy PDFs
# (docs/) are ingested separately below, also matching main_simulation.py.
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
        metadata={"source": "web_chat", "user_id": "alum_002"},
    ),
]

# Comfortably above the seeded catalog's real size (75 items as of
# db/seed_postgres.sql) without hardcoding an exact count that would go
# stale the moment the catalog grows.
_CATALOG_INGEST_LIMIT = 500


def _build_catalog_products(ecommerce_client: EcommerceClient) -> list[RawProductRecord]:
    """Real catalog items, shaped for ingest_product_catalog() - replaces
    two hardcoded phantom products (PROD_LAP_001/PROD_HEAD_002) that used to
    be RAG-ingested here but were never real items.db rows: ProductRecommendationAgent's
    RAG-fallback path resolves a RAG match's product_id via
    EcommerceClient.get_item() before recommending it (services/agents/
    product_recommendation_agent.py), so a RAG-indexed product that isn't a
    real row can only ever end in "no recommendation found" - and, worse,
    can win a nearest-neighbour match away from a real, relevant item on a
    topically-similar query (e.g. "laptop" matching the phantom "gaming
    laptop" text ahead of the real "Laptop Cooling Pad" row), turning a
    resolvable recommendation into a dead end. Indexing the real catalog
    instead makes every RAG match resolvable and removes that collision
    risk entirely.
    """
    items = ecommerce_client.search_items(limit=_CATALOG_INGEST_LIMIT)
    return [
        RawProductRecord(
            product_id=item["item_id"],
            raw_description=item.get("description") or item["name"],
            specs={"category": item["category"]} if item.get("category") else {},
            reviews=[],
            price=str(item["price"]),
        )
        for item in items
    ]


@lru_cache
def get_pii_masker() -> PIIMasker:
    return PIIMasker()


@lru_cache
def get_llm_service() -> LLMInferenceService:
    return LLMInferenceService()


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService(get_llm_service())


@lru_cache
def get_ecommerce_client() -> EcommerceClient:
    return EcommerceClient()


@lru_cache
def get_classifier_client() -> ClassifierClient:
    return ClassifierClient()


@lru_cache
def get_data_pipeline() -> DataIngestionPipeline:
    return DataIngestionPipeline(get_pii_masker(), get_llm_service(), get_rag_service(), get_classifier_client())

@lru_cache
def get_agents() -> dict[str, BaseAgent]:
    """The canonical agent registry - the only place agent name -> instance
    is decided. This dict's keys are load-bearing, not just labels:
    services/orchestrator.py dispatches sub-tasks by looking a
    decomposer-supplied name up in this exact dict (falling back to
    GeneralPurposeAgent on a miss), and services/llm_inference.py's
    call_task_decomposer() is handed `list(self.agents)` from here so the
    LLM is only ever offered names that actually resolve.

    Each key here MUST match the literal string that agent's own __init__
    passes to BaseAgent.__init__ (e.g. OrderTrackingAgent's
    `super().__init__("OrderTrackingAgent", ...)`) - that string becomes
    self.name, which flows into every StructuredAgentResult.agent_name this
    agent returns and ultimately the customer-facing
    ChatbotResponse.agent_invoked. Nothing enforces this match
    automatically: renaming a key here without updating the matching
    agent's __init__ (or vice versa) won't error - dispatch still works
    because it's driven by this dict, not by self.name - but every log
    line and the customer-facing response would keep reporting the old
    name. main_simulation.py constructs the same four agents against the
    same four name strings for the CLI entry point; keep both in sync by
    hand if an agent is ever added, renamed, or removed here.
    """
    deps = (get_llm_service(), get_rag_service(), get_ecommerce_client())
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
    return AgentOrchestratorService(get_llm_service(), get_pii_masker(), get_agents(), get_classifier_client())


def warm_up_services() -> None:
    """Eagerly construct every singleton and seed the RAG store at startup
    (called from main.py's lifespan handler), instead of lazily on first
    request."""
    logger.info("[API] Warming up services...")
    get_orchestrator()  # transitively constructs everything else
    pipeline = get_data_pipeline()

    # main.py's lifespan handler awaits this directly with nothing else
    # guarding it - an exception here fails FastAPI's startup entirely and
    # the API never starts serving requests at all. Each ingestion step
    # already skips individual chunks/products/files on an embedding
    # failure rather than raising (see data_pipeline.py), but this is a
    # deliberate second layer: even an unrelated/unexpected failure in one
    # ingestion step (a DB hiccup, a bug) shouldn't block the other two, or
    # take down the whole API - a container that boots with a partially (or
    # even completely) empty RAG store is recoverable; one that never boots
    # isn't.
    for step_name, step in (
        ("customer conversations", lambda: pipeline.ingest_customer_conversations(_SAMPLE_CONVERSATIONS)),
        ("product catalog", lambda: pipeline.ingest_product_catalog(_build_catalog_products(get_ecommerce_client()))),
        ("PDF documents", lambda: pipeline.ingest_pdf_documents(docs_folder="docs", source_type="customer_policy")),
    ):
        try:
            step()
        except Exception:
            logger.critical(
                "[API] Warm-up ingestion step '%s' failed - continuing startup with RAG store incomplete for this step",
                step_name, exc_info=True,
            )
    logger.info("[API] Service warm-up complete.")
