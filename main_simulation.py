# main_simulation.py
"""
End-to-end simulation of the AI Agentic Customer Support Platform.

What this script does:
  1. Initialises all services (PII masker, LLM inference, RAG / ChromaDB,
     e-commerce API client, data pipeline, specialised agents, orchestrator).
  2. Runs the data ingestion cycle (conversations, product catalogue, policies).
  3. Simulates five realistic customer interactions covering every agent path.

Run:
    python main_simulation.py

Environment:
    Copy .env.example to .env and fill in ANTHROPIC_API_KEY.
    Without an API key the system runs in mock-LLM mode (all logic still executes).
"""

import logging
import os
import uuid
from datetime import datetime

# ── Common models ─────────────────────────────────────────────────────────────
from common.models import (
    CustomerQuery,
    RawCustomerConversation,
    RawProductRecord,
)

# ── Clients ───────────────────────────────────────────────────────────────────
from clients.ecommerce_api_client import MockECommerceAPIClient

# ── Services ──────────────────────────────────────────────────────────────────
from services.pii_masker import PIIMasker
from services.llm_inference import LLMInferenceService
from services.rag import RAGService
from services.data_pipeline import DataIngestionPipeline
from services.orchestrator import AgentOrchestratorService
from services.evaluation import EvaluationService, latency_timer

# ── Agents ────────────────────────────────────────────────────────────────────
from services.agents.order_tracking_agent import OrderTrackingAgent
from services.agents.product_recommendation_agent import ProductRecommendationAgent
from services.agents.general_purpose_agent import GeneralPurposeAgent
from services.agents.returns_agent import ReturnsAgent
from services.agents.escalation_agent import EscalationAgent

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_separator(title: str = "") -> None:
    line = "─" * 80
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(f"{line}")
    else:
        print(line)


def run_interaction(
    orchestrator: AgentOrchestratorService,
    query: CustomerQuery,
    latencies: list[float] | None = None,
    statuses: list[str] | None = None,
) -> None:
    """Run a single customer interaction, pretty-print the result, and
    optionally record its latency + resolution status for evaluation."""
    print(f"\n>>> Customer ({query.user_id}): \"{query.text}\"")
    print(f"    Session: {query.session_id}")

    with latency_timer() as timer:
        response = orchestrator.handle_customer_query(query)

    print(f"\n<<< Support Bot: \"{response.response_text}\"")
    print(f"    Agent invoked : {response.agent_invoked}")
    print(f"    Confidence    : {response.confidence_score:.2f}")
    print(f"    Latency       : {timer.elapsed_seconds:.3f}s")
    print_separator()

    if latencies is not None:
        latencies.append(timer.elapsed_seconds)
    if statuses is not None:
        # Treat confidence 0.0 (our escalation signal) as a non-"success" outcome
        statuses.append("escalation" if response.confidence_score == 0.0 else "success")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_separator("AI Agentic Customer Support Platform — System Initialisation")

    # ── 1. Core services ──────────────────────────────────────────────────────
    pii_masker = PIIMasker()                                # regex mode by default
    llm_service = LLMInferenceService()                      # LangChain LCEL chains (or mock fallback)
    rag_service = RAGService()                               # LangChain Chroma vectorstore
    ecommerce_client = MockECommerceAPIClient()              # Mock CRM / Order DB

    logger.info("Core services initialised.")

    # ── 2. Data ingestion pipeline ────────────────────────────────────────────
    pipeline = DataIngestionPipeline(pii_masker, llm_service, rag_service)

    print_separator("Data Ingestion Cycle")

    # ── 2a. Customer conversations ────────────────────────────────────────────
    raw_conversations = [
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
            text="I love my new laptop! Is there a warranty? My phone is 123-456-7890.",
            metadata={"source": "web_chat", "user_id": "cust_002"},
        ),
    ]
    cleaned_convs = pipeline.ingest_customer_conversations(raw_conversations)
    logger.info(
        "Conversations ingested. Sample masked text: '%s'",
        cleaned_convs[0].cleaned_text[:60],
    )

    # ── 2b. Product catalogue ─────────────────────────────────────────────────
    raw_products = [
        RawProductRecord(
            product_id="PROD_LAP_001",
            raw_description=(
                "High-performance gaming laptop with an Intel Core i7 processor, "
                "16GB DDR5 RAM, and a 1TB NVMe SSD. Stunning 144Hz QHD display "
                "with RGB keyboard. Ideal for gaming and creative professionals."
            ),
            specs={"CPU": "Intel Core i7", "RAM": "16GB DDR5", "Storage": "1TB NVMe SSD",
                   "Display": "144Hz QHD"},
            reviews=[
                "Great product! Fast and reliable.",
                "Screen is amazing, colours are vivid.",
                "Runs hot under heavy load, but overall excellent.",
                "Fast delivery. Jane Smith recommended it to me!",  # PII in review
            ],
            price="$1,200.00",
        ),
        RawProductRecord(
            product_id="PROD_HEAD_002",
            raw_description=(
                "Premium noise-cancelling headphones for immersive audio. "
                "Comfortable over-ear design with 20-hour battery life and "
                "foldable build. Ideal for travel and remote work."
            ),
            specs={"Color": "Midnight Black", "Battery": "20 hours", "Connectivity": "Bluetooth 5.3"},
            reviews=[
                "Awesome sound quality!",
                "Very comfortable for long sessions.",
                "Noise cancellation is top-notch.",
            ],
            price="$250.00",
        ),
        RawProductRecord(
            product_id="PROD_WATCH_003",
            raw_description=(
                "Smart fitness watch with heart-rate monitor, GPS, sleep tracking, "
                "and 7-day battery. Water-resistant up to 50m. Compatible with iOS and Android."
            ),
            specs={"Battery": "7 days", "Water resistance": "50m", "OS": "iOS & Android"},
            reviews=["Great for running!", "Battery life is impressive.", "GPS lock is quick."],
            price="$199.99",
        ),
    ]
    cleaned_products = pipeline.ingest_product_catalog(raw_products)
    logger.info(
        "Products ingested. Catalogue size: %d chunks in RAG.",
        rag_service.collection_size(),
    )

    # ── 2c. Policy documents ──────────────────────────────────────────────────
    policies = [
        {
            "id": "pol_returns_001",
            "title": "Return & Refund Policy",
            "content": (
                "Our return policy allows customers to return most items within 30 days "
                "of purchase, provided they are in original condition and packaging. "
                "To initiate a return, contact support with your order ID. "
                "Refunds are processed within 5-7 business days to the original payment method. "
                "Electronics must be returned within 15 days. Opened software is non-refundable. "
                "Free return shipping is provided for defective items."
            ),
        },
        {
            "id": "pol_warranty_001",
            "title": "Warranty Policy",
            "content": (
                "All products come with a minimum 1-year manufacturer warranty. "
                "Laptops and electronics carry a 2-year warranty covering manufacturing defects. "
                "Warranty does not cover accidental damage, water damage, or unauthorised repairs. "
                "To claim warranty, contact support with proof of purchase and a description of the issue. "
                "Replacement units are dispatched within 3-5 business days upon approval."
            ),
        },
        {
            "id": "pol_shipping_001",
            "title": "Shipping Policy",
            "content": (
                "Standard shipping takes 5-7 business days. Express shipping (2-3 days) is available "
                "at checkout for an additional fee. Free standard shipping on orders over $50. "
                "International shipping is available to 40+ countries. "
                "Once shipped, you will receive a tracking number via email. "
                "Estimated delivery dates are shown at checkout."
            ),
        },
    ]
    pipeline.ingest_policy_documents(policies)
    logger.info(
        "Policies ingested. Total RAG collection size: %d chunks.",
        rag_service.collection_size(),
    )

    # ── 2d. Optional: real-world Twitter customer-support data ────────────────
    # Drop a "Customer Support on Twitter"-style CSV at this path to enrich
    # the knowledge base with real inbound/outbound conversation pairs.
    twitter_csv_path = "sample.csv"
    if os.path.exists(twitter_csv_path):
        twitter_convs = pipeline.ingest_twitter_support_csv(
            twitter_csv_path, max_conversations=50
        )
        logger.info(
            "Twitter support data ingested: %d conversation(s). "
            "Total RAG collection size: %d chunks.",
            len(twitter_convs), rag_service.collection_size(),
        )
    else:
        logger.info(
            "No Twitter support CSV found at '%s' — skipping (optional data source).",
            twitter_csv_path,
        )

    # ── 2d. Synthetic query generation ────────────────────────────────────────
    synthetic = pipeline.generate_synthetic_queries([
        "Where is my shipment?",
        "How do I return an item?",
        "Suggest a gift for a gamer.",
    ])
    logger.info("Generated %d synthetic training query variants.", len(synthetic))

    # ── 3. Specialised agents ─────────────────────────────────────────────────
    print_separator("Initialising Agents")

    agent_deps = (llm_service, rag_service, ecommerce_client, pii_masker)
    agents = {
        "OrderTrackingAgent":        OrderTrackingAgent(*agent_deps),
        "ProductRecommendationAgent": ProductRecommendationAgent(*agent_deps),
        "GeneralPurposeAgent":       GeneralPurposeAgent(*agent_deps),
        "ReturnsAgent":              ReturnsAgent(*agent_deps),
        "EscalationAgent":           EscalationAgent(*agent_deps),
    }
    logger.info("Registered agents: %s", list(agents.keys()))

    # ── 4. Orchestrator ───────────────────────────────────────────────────────
    orchestrator = AgentOrchestratorService(llm_service, pii_masker, agents)
    logger.info("Orchestrator ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Simulate customer interactions
    # ─────────────────────────────────────────────────────────────────────────
    print_separator("SYSTEM READY — Simulating Customer Interactions")

    # Track per-turn latency and resolution status for the evaluation report
    turn_latencies: list[float] = []
    turn_statuses: list[str] = []

    # Interaction 1: Order status check (OrderTrackingAgent)
    session_1 = f"session_{uuid.uuid4().hex[:8]}"
    run_interaction(
        orchestrator,
        CustomerQuery(
            session_id=session_1,
            user_id="cust_001",
            text="Hi, I'd like to check my order status for order 12345.",
        ),
        turn_latencies, turn_statuses,
    )

    # Interaction 2: Follow-up in same session — different order (OrderTrackingAgent)
    run_interaction(
        orchestrator,
        CustomerQuery(
            session_id=session_1,          # Same session — history is maintained
            user_id="cust_001",
            text="Actually, my name is Jane Smith. What about order 54321, is that shipped?",
        ),
        turn_latencies, turn_statuses,
    )

    # Interaction 3: Product recommendation (ProductRecommendationAgent)
    session_2 = f"session_{uuid.uuid4().hex[:8]}"
    run_interaction(
        orchestrator,
        CustomerQuery(
            session_id=session_2,
            user_id="cust_002",
            text="Can you recommend a good laptop for gaming? My budget is around $1200.",
        ),
        turn_latencies, turn_statuses,
    )

    # Interaction 4: Policy query with PII (GeneralPurposeAgent)
    session_3 = f"session_{uuid.uuid4().hex[:8]}"
    run_interaction(
        orchestrator,
        CustomerQuery(
            session_id=session_3,
            user_id="cust_003",
            text="What's your return policy? My email is john.doe@example.com.",
        ),
        turn_latencies, turn_statuses,
    )

    # Interaction 5: General / unrecognised query (GeneralPurposeAgent)
    session_4 = f"session_{uuid.uuid4().hex[:8]}"
    run_interaction(
        orchestrator,
        CustomerQuery(
            session_id=session_4,
            user_id="cust_004",
            text="Tell me about your company's history and founding story.",
        ),
        turn_latencies, turn_statuses,
    )

    # ── Show conversation history for session_1 ───────────────────────────────
    print_separator("Session History Demo (session_1)")
    history = orchestrator.get_session_history(session_1)
    for turn in history:
        label = "Customer" if turn["role"] == "user" else "Bot"
        print(f"  [{label}]: {turn['content'][:100]}{'…' if len(turn['content']) > 100 else ''}")

    print_separator("Simulation Complete")
    print(f"\n  RAG collection size : {rag_service.collection_size()} chunks")
    print(f"  Sessions handled    : 4")
    print(f"  Synthetic queries   : {len(synthetic)}")

    # ── 6. Evaluation Framework report (Response Speed + Workflow Completion) ─
    # These two dimensions need no ground-truth labels, so we can compute them
    # directly from this simulation run. The remaining dimensions (Context
    # Recall/Precision, Faithfulness, Answer Relevance, BERTScore F1,
    # Hallucination Rate) need a labelled test set (questions + reference
    # answers + retrieved contexts) — see services/evaluation.py's
    # `build_report()` for the full-framework entry point once you have one.
    print_separator("Evaluation Framework — Live-Measurable Dimensions")
    evaluator = EvaluationService()
    speed_result = evaluator.evaluate_response_speed(turn_latencies)
    completion_result = evaluator.evaluate_workflow_completion(turn_statuses)
    for result in (speed_result, completion_result):
        status = "N/A" if result.passed is None else ("PASS" if result.passed else "FAIL")
        value_str = "N/A" if result.value is None else f"{result.value:.3f}"
        print(f"  [{status:4}] {result.name:<24} value={value_str:<8} target={result.comparator}{result.target}")
    print()

