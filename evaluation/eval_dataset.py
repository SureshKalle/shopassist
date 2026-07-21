# evaluation/eval_dataset.py
"""
Hand-labeled evaluation set for shopassist. Two kinds of cases:

1. ROUTING_CASES - a (query, expected_agent) pair, used to score
   `LLMInferenceService.call_router()` (via the orchestrator's actual
   agent_invoked) with sklearn's F1/precision/recall - see metrics.py's
   `compute_routing_metrics()`.

2. RAG_CASES - a (query, reference_keywords) pair for queries expected to
   be answered via `GeneralPurposeAgent` / `MockRAGService`. reference_
   keywords are the terms a *correct, grounded* answer should touch on -
   used as a cheap proxy ground-truth for context precision/recall
   (there's no full reference-answer corpus for this project, so keyword
   coverage stands in for it - see metrics.py's docstring for the caveat).

Both use the seeded SQLite dev DB (db/seed_sqlite.sql) business-key IDs
(alum-1001 etc.) and ord-1001-style order IDs, matching main_simulation.py's
convention - swap in real IDs from your seed data as needed.

Extend these lists directly; nothing else needs to change to add a case.
"""
from typing import Dict, List, TypedDict, Optional


class RoutingCase(TypedDict):
    query: str
    user_id: str
    expected_agent: str


class RagCase(TypedDict):
    query: str
    user_id: str
    reference_keywords: List[str]
    reference_answer: Optional[str]  # optional; used only for an eyeball diff in the report


# --- 1. Intent routing cases -------------------------------------------------
# expected_agent must match one of the keys in main_simulation.py's `agents` dict.
ROUTING_CASES: List[RoutingCase] = [
    {"query": "Hi, I'd like to check my order status for order ord-1001.", "user_id": "alum-1001", "expected_agent": "OrderTrackingAgent"},
     {"query": "Where is my stuff? Order ord-1002 please.", "user_id": "alum-1002", "expected_agent": "OrderTrackingAgent"},
     {"query": "I want to cancel order ord-1003.", "user_id": "alum-1003", "expected_agent": "OrderTrackingAgent"},
     {"query": "Can you recommend a good laptop for gaming?", "user_id": "alum-1001", "expected_agent": "ProductRecommendationAgent"},
     {"query": "What should I buy for a birthday gift?", "user_id": "alum-1002", "expected_agent": "ProductRecommendationAgent"},
     {"query": "I want to return a product and get a refund.", "user_id": "alum-1003", "expected_agent": "ProductRecommendationAgent"},
     {"query": "What is your return policy?", "user_id": "alum-1001", "expected_agent": "GeneralPurposeAgent"},
     {"query": "Tell me about your company's history.", "user_id": "alum-1002", "expected_agent": "GeneralPurposeAgent"},
     {"query": "Do you ship internationally?", "user_id": "alum-1003", "expected_agent": "GeneralPurposeAgent"},
]


# --- 2. RAG / RAGAS-style cases ---------------------------------------------
# These are expected to land on GeneralPurposeAgent, which only calls
# RAGService.query_knowledge_base() - no EcommerceClient/DB dependency, so
# these are safe to run without seeding extra data beyond what run_eval.py
# ingests itself (see its SAMPLE_DOCS).
RAG_CASES: List[RagCase] = [
    {
        "query": "What is your return policy?",
        "user_id": "alum-1001",
        "reference_keywords": ["return", "30 days", "original packaging"],
        "reference_answer": "Items can be returned within 30 days of purchase if they are in original packaging.",
    },
    {
        "query": "What are the features of your high-end gaming laptops?",
        "user_id": "alum-1002",
        "reference_keywords": ["i7", "16GB", "SSD", "gaming"],
        "reference_answer": "The gaming laptop has an i7 processor, 16GB RAM, and a 1TB SSD, ideal for gaming.",
    },
    {
        "query": "Do you offer a warranty on electronics?",
        "user_id": "alum-1003",
        "reference_keywords": ["warranty", "1 year", "electronics"],
        "reference_answer": "Electronics come with a 1-year manufacturer warranty.",
    },
]


def get_routing_dataset() -> List[RoutingCase]:
    return ROUTING_CASES


def get_rag_dataset() -> List[RagCase]:
    return RAG_CASES
