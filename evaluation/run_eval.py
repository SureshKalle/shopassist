#!/usr/bin/env python3
# evaluation/run_eval.py
"""
Runnable evaluation harness for shopassist - built the same way
main_simulation.py wires up services/agents/orchestrator, so this needs no
running API server (no uvicorn, no docker) - just `python evaluation/run_eval.py`
from the project root.

What it does:
  1. Rebuilds the local SQLite dev DB (db.init_db.build_db(), same as
     main_simulation.py) so OrderTrackingAgent/ProductRecommendationAgent
     cases resolve against real seeded rows.
  2. Ingests a small sample document set into MockRAGService (policy/product
     text) via DataIngestionPipeline, so GeneralPurposeAgent has something to
     retrieve for evaluation.RAG_CASES.
  3. Runs evaluation.eval_dataset.ROUTING_CASES through the real
     AgentOrchestratorService and scores agent_invoked vs expected_agent
     with evaluation.metrics.compute_routing_metrics() (sklearn F1).
  4. Runs evaluation.eval_dataset.RAG_CASES the same way, additionally
     capturing what MockRAGService actually retrieved (via a thin wrapper
     around query_knowledge_base - the orchestrator's response doesn't
     expose retrieved contexts, so this is instrumented here rather than
     changing orchestrator.py's public contract) and scores the result with
     evaluation.metrics.compute_ragas_style_metrics().
  5. Prints both reports and writes a combined JSON report to
     evaluation/eval_report.json for CI/history tracking.

Requires scikit-learn and numpy in addition to requirements.txt - see
requirements-eval.txt alongside this file.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

# Allow running as `python evaluation/run_eval.py` from the project root
# without needing the project installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.init_db import build_db
from common.models import CustomerQuery, RawCustomerConversation, RawProductRecord
from clients.ecommerce_api_client import EcommerceClient
from services.pii_masker import PIIMasker
from services.llm_inference import LLMInferenceService
from services.classifier_client import ClassifierClient
from services.rag import RAGService
from services.data_pipeline import DataIngestionPipeline
from services.agents.order_tracking_agent import OrderTrackingAgent
from services.agents.product_recommendation_agent import ProductRecommendationAgent
from services.agents.general_purpose_agent import GeneralPurposeAgent
from services.agents.escalation_agent import EscalationAgent
from services.orchestrator import AgentOrchestratorService

from evaluation.eval_dataset import get_rag_dataset, get_routing_dataset
from evaluation.metrics import compute_ragas_style_metrics, compute_routing_metrics
from evaluation.report import build_and_write_report

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING").upper(),  # quiet by default so the report is readable
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).resolve().parent / "eval_report.json"
REPORT_HTML_PATH = Path(__file__).resolve().parent / "eval_report.html"


# Sample docs so GeneralPurposeAgent/MockRAGService has something to retrieve
# for RAG_CASES - independent of main_simulation.py's own sample data, kept
# local to this script so the eval set and the ingested corpus stay in sync.
SAMPLE_CUSTOMER_CONVERSATIONS = [
    RawCustomerConversation(
        id="eval_conv_001",
        text="Our return policy allows returns within 30 days of purchase provided the item is in its original packaging.",
        metadata={"source": "policy_doc", "user_id": "system"},
    ),
]

SAMPLE_PRODUCT_CATALOG = [
    RawProductRecord(
        product_id="EVAL_PROD_LAP_001",
        raw_description="High-performance gaming laptop with an i7 processor, 16GB RAM, and a 1TB SSD.",
        specs={"CPU": "i7", "RAM": "16GB", "Storage": "1TB SSD"},
        reviews=["Great gaming laptop!", "Screen is amazing."],
        price="₹1200.00",
    ),
    RawProductRecord(
        product_id="EVAL_PROD_ELEC_002",
        raw_description="All electronics purchased from us include a 1 year manufacturer warranty for peace of mind.",
        specs={"Warranty": "1 year"},
        reviews=["Warranty support was excellent."],
        price="₹500.00",
    ),
]


def build_stack():
    """Wire up the same services/agents/orchestrator as main_simulation.py."""
    build_db()

    pii_masker = PIIMasker()
    llm_inference_service = LLMInferenceService()
    rag_service = RAGService(llm_inference_service)
    ecommerce_api_client = EcommerceClient()
    classifier_client = ClassifierClient()

    data_pipeline = DataIngestionPipeline(pii_masker, llm_inference_service, rag_service, classifier_client)
    data_pipeline.ingest_customer_conversations(SAMPLE_CUSTOMER_CONVERSATIONS)
    data_pipeline.ingest_product_catalog(SAMPLE_PRODUCT_CATALOG)

    agents = {
        "OrderTrackingAgent": OrderTrackingAgent(llm_inference_service, rag_service, ecommerce_api_client, pii_masker),
        "ProductRecommendationAgent": ProductRecommendationAgent(llm_inference_service, rag_service, ecommerce_api_client, pii_masker),
        "GeneralPurposeAgent": GeneralPurposeAgent(llm_inference_service, rag_service, ecommerce_api_client, pii_masker),
        "EscalationAgent": EscalationAgent(llm_inference_service, rag_service, ecommerce_api_client, pii_masker),
    }
    orchestrator = AgentOrchestratorService(llm_inference_service, pii_masker, agents, classifier_client)
    return orchestrator, rag_service, llm_inference_service


def _wrap_rag_service_to_capture_contexts(rag_service: RAGService) -> Dict[str, List[Any]]:
    """MockRAGService.query_knowledge_base() results never make it back out
    through ChatbotResponse - GeneralPurposeAgent consumes them internally
    (services/agents/general_purpose_agent.py). To score context_precision/
    context_recall we need to see what was actually retrieved for a given
    customer turn, so this wraps the instance method to stash the most
    recent call's results in `captured["docs"]`, without touching
    orchestrator.py's or the agent's public contract.
    """
    captured: Dict[str, List[Any]] = {"docs": []}
    original = rag_service.query_knowledge_base

    def _wrapped(*args, **kwargs):
        docs = original(*args, **kwargs)
        captured["docs"] = docs
        return docs

    rag_service.query_knowledge_base = _wrapped  # type: ignore[method-assign]
    return captured


def run_routing_eval(orchestrator: AgentOrchestratorService) -> Dict[str, Any]:
    y_true, y_pred = [], []
    rows = []
    for case in get_routing_dataset():
        session_id = f"eval_routing_{uuid.uuid4().hex[:8]}"
        query = CustomerQuery(session_id=session_id, user_id=case["user_id"], text=case["query"])
        response = orchestrator.handle_customer_query(query)
        y_true.append(case["expected_agent"])
        y_pred.append(response.agent_invoked)
        rows.append({
            "query": case["query"],
            "expected_agent": case["expected_agent"],
            "actual_agent": response.agent_invoked,
            "correct": case["expected_agent"] == response.agent_invoked,
        })
    metrics = compute_routing_metrics(y_true, y_pred)
    return {"rows": rows, "metrics": metrics}


def run_rag_eval(orchestrator: AgentOrchestratorService, captured: Dict[str, List[Any]], llm_inference_service) -> Dict[str, Any]:
    cases_for_metrics = []
    for case in get_rag_dataset():
        session_id = f"eval_rag_{uuid.uuid4().hex[:8]}"
        captured["docs"] = []  # reset before each call so we only see this turn's retrieval
        query = CustomerQuery(session_id=session_id, user_id=case["user_id"], text=case["query"])
        response = orchestrator.handle_customer_query(query)

        contexts = [doc.content for doc in captured["docs"]]
        if not contexts:
            logger.warning(
                "RAG eval: query %r retrieved 0 contexts. If this happens for "
                "every case, check rag_service.faiss_index.ntotal (>0?) and "
                "compare doc/query embedding dimensions — a mismatch there "
                "silently drops documents during ingestion, not just this query.",
                case["query"],
            )
        cases_for_metrics.append({
            "question": case["query"],
            "answer": response.response_text,
            "contexts": contexts,
            "reference_keywords": case["reference_keywords"],
        })
    return compute_ragas_style_metrics(cases_for_metrics, llm_inference_service)


def print_routing_report(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("INTENT ROUTING - F1 REPORT")
    print("=" * 72)
    for row in result["rows"]:
        mark = "PASS" if row["correct"] else "FAIL"
        print(f"[{mark}] '{row['query'][:60]}' -> expected={row['expected_agent']} actual={row['actual_agent']}")
    m = result["metrics"]
    if "error" in m:
        print(m["error"])
        return
    print(f"\naccuracy={m['accuracy']}  f1_macro={m['f1_macro']}  f1_weighted={m['f1_weighted']}")
    print("\nPer-class:")
    for label, stats in m["per_class"].items():
        print(f"  {label:<28} precision={stats['precision']:<6} recall={stats['recall']:<6} f1={stats['f1']:<6} support={stats['support']}")
    print("\n" + m["classification_report"])


def print_rag_report(result: Dict[str, Any]) -> None:
    print("=" * 72)
    print("RAG / RAGAS-STYLE REPORT")
    print("=" * 72)
    for row in result["rows"]:
        print(f"\nQ: {row['question']}")
        print(f"A: {row['answer'][:150]}")
        print(f"   contexts retrieved: {row['n_contexts']}")
        print(f"   context_precision={row['context_precision']}  context_recall={row['context_recall']}  "
              f"faithfulness={row['faithfulness']}  answer_relevancy={row['answer_relevancy']}")
    agg = result["aggregates"]
    print("\n" + "-" * 72)
    print(f"Averages over {agg['n_cases']} case(s):")
    print(f"  avg_context_precision = {agg['avg_context_precision']}")
    print(f"  avg_context_recall    = {agg['avg_context_recall']}")
    print(f"  avg_faithfulness      = {agg['avg_faithfulness']}")
    print(f"  avg_answer_relevancy  = {agg['avg_answer_relevancy']}")
    print("=" * 72)


def main() -> None:
    print("--- Building service stack for evaluation ---")
    orchestrator, rag_service, llm_inference_service = build_stack()
    captured = _wrap_rag_service_to_capture_contexts(rag_service)

    print("\n--- Running intent routing evaluation ---")
    routing_result = run_routing_eval(orchestrator)
    print_routing_report(routing_result)

    print("\n--- Running RAG / RAGAS-style evaluation ---")
    rag_result = run_rag_eval(orchestrator, captured, llm_inference_service)
    print_rag_report(rag_result)

    combined_report = {"routing": routing_result, "rag": rag_result}
    REPORT_PATH.write_text(json.dumps(combined_report, indent=2, default=str))
    print(f"\nFull report written to {REPORT_PATH}")

    html_path = build_and_write_report(combined_report, REPORT_HTML_PATH)
    print(f"Rich HTML report written to {html_path} - open it directly in a browser.")


if __name__ == "__main__":
    main()
