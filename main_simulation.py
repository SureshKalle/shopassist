# main_simulation.py
"""
CLI walkthrough of the whole shopassist pipeline, no HTTP server required.
Rebuilds the local SQLite dev DB, wires up every service/agent exactly like
api/dependencies.py does for the FastAPI app, ingests a small sample dataset,
then runs one hardcoded customer interaction end to end (four more are
included commented-out below, covering product recommendations, PII masking,
and escalation).

Useful for a first read of the request flow without needing curl/Postman, and
for quick manual testing of an agent/service change without going through the
API layer at all.
"""
import logging
import os
import uuid

from db.init_db import build_db
from common.models import CustomerQuery, RawCustomerConversation, RawProductRecord
from clients.ecommerce_api_client import EcommerceClient
from services.pii_masker import PIIMasker
from services.llm_inference import LLMInferenceService
from services.classifier_client import ClassifierClient
from services.rag import MockRAGService
from services.data_pipeline import DataIngestionPipeline
from services.agents.order_tracking_agent import OrderTrackingAgent
from services.agents.product_recommendation_agent import ProductRecommendationAgent
from services.agents.general_purpose_agent import GeneralPurposeAgent
from services.agents.escalation_agent import EscalationAgent
from services.orchestrator import AgentOrchestratorService

# --- Langfuse Integration Start ---
from common.langfuse_config import initialize_langfuse_client, get_langfuse_client_instance
# No need to import `get_client` from `langfuse` itself if we only use `get_langfuse_client_instance`
# from langfuse import get_client as get_langfuse_context_client # Alias to avoid conflict if `get_client` is used elsewhere
# --- Langfuse Integration End ---

# Same LOG_LEVEL convention api/main.py uses - the services/ layer logs via
# the standard `logging` module now (see e.g. clients/ecommerce_api_client.py),
# so this is required for those log lines to actually show up when running
# this script directly; without a configured handler, Python's root logger
# silently drops everything below WARNING.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# --- Main Execution Flow (Simulates System Startup and Customer Interactions) ---

if __name__ == "__main__":
    print("--- Initializing Chatbot System Components ---")

    # --- Langfuse Integration Start: Initialize Client ---
    print("--- Initializing Langfuse Client ---")
    initialize_langfuse_client()
    print("--- Langfuse Client Initialized ---")
    # --- Langfuse Integration End ---

    # 0. Initialize/Rebuild the local dev DB (schema + seed data)
    build_db()
    
    # 1. Initialize Core Services
    pii_masker = PIIMasker()
    llm_inference_service = LLMInferenceService()
    rag_service = MockRAGService(llm_inference_service) # RAG needs LLM for embeddings
    ecommerce_api_client = EcommerceClient()
    classifier_client = ClassifierClient() # sentiment; fails soft if not started separately (shopassist-model)

    # 2. Initialize Data Ingestion Pipeline
    data_pipeline = DataIngestionPipeline(pii_masker, llm_inference_service, rag_service, classifier_client)

    # --- SIMULATE DATA PREPARATION & INGESTION ---
    print("\n--- Running Data Preparation & Ingestion Cycle ---")
    
    # Raw Customer Conversations
    raw_customer_conversations = [
        RawCustomerConversation(id="conv_001", text="Hi, my name is John Doe, and I want to know about my order 12345.", metadata={"source": "twitter", "user_id": "jd_123"}),
        RawCustomerConversation(id="conv_002", text="Can you help me with a return for product X? My email is john.doe@example.com.", metadata={"source": "web_form", "user_id": "jd_123"}),
        RawCustomerConversation(id="conv_003", text="I love my new laptop! Is there a warranty?", metadata={"source": "web_chat", "user_id": "alum_002"}),
    ]
    cleaned_convs = data_pipeline.ingest_customer_conversations(raw_customer_conversations)
    print(f"\nSample Cleaned Conversation for LLM Fine-tuning: '{cleaned_convs[0].cleaned_text[:50]}...'")

    # Raw E-commerce Product Catalog
    raw_product_catalog = [
        RawProductRecord(product_id="PROD_LAP_001", raw_description="High-performance gaming laptop with an i7 processor, 16GB RAM, and a 1TB SSD. Stunning display and RGB keyboard.", specs={"CPU": "i7", "RAM": "16GB", "Storage": "1TB SSD"}, reviews=["Great product!", "Fast delivery.", "Screen is amazing!"], price="₹1200.00"),
        RawProductRecord(product_id="PROD_HEAD_002", raw_description="Premium noise-cancelling headphones for immersive audio. Comfortable earcups and 20-hour battery life.", specs={"Color": "Black", "Battery": "20h"}, reviews=["Awesome sound!", "John Doe found them comfy and fit perfectly."], price="₹250.00"),
    ]
    cleaned_products = data_pipeline.ingest_product_catalog(raw_product_catalog)
    print(f"\nSample Cleaned Product Description for LLM Fine-tuning: '{cleaned_products[0].clean_description[:50]}...'")
    print(f"Product RAG vector DB now contains {len(rag_service.vector_db)} documents from initial ingestion.")

    # Synthetic E-commerce Queries
    synthetic_queries = data_pipeline.generate_synthetic_queries(["Where is my shipment?", "Suggest a gift.", "How do I return an item?"])
    print(f"\nSynthetic queries (Sample for LLM training): {synthetic_queries[0]}, {synthetic_queries[1]}")
    print("--- Data Preparation & Ingestion Cycle Complete ---")

    # 3. Initialize Specialized Agents
    agents = {
        "OrderTrackingAgent": OrderTrackingAgent(llm_inference_service, rag_service, ecommerce_api_client, pii_masker),
        "ProductRecommendationAgent": ProductRecommendationAgent(llm_inference_service, rag_service, ecommerce_api_client, pii_masker),
        "GeneralPurposeAgent": GeneralPurposeAgent(llm_inference_service, rag_service, ecommerce_api_client, pii_masker),
        "EscalationAgent": EscalationAgent(llm_inference_service, rag_service, ecommerce_api_client, pii_masker),
        # You would add "ReturnsAgent" and other agents here
    }
    print("\nInitialized Specialized AI Agents.")

    # 4. Initialize Agent Orchestrator
    orchestrator = AgentOrchestratorService(llm_inference_service, pii_masker, agents, classifier_client)
    print("Initialized Agent Orchestrator Service.")

    print("\n--- SYSTEM READY: Simulating Customer Interactions ---\n")

    # --- SIMULATE CUSTOMER INTERACTIONS ---
    #
    # NOTE on user_id="alum-1001" below: CustomerQuery.user_id is the one
    # identifier end to end now (see common/models.py) - the same value
    # shopassist-client sends at login, used directly for order/customer DB
    # lookups (customers.user_id, db/README.md). There's no real per-customer
    # login here, so every interaction below is hardcoded to the same seeded
    # customer - see db/seed_sqlite.sql for the full list of valid user_id
    # values (alum-1001 = Aarav Sharma, alum-1002 = Ananya Iyer, ...) - purely
    # to exercise the downstream order-lookup/recommendation logic end to
    # end. This is a simulation-only stand-in kept for reference;
    # api/routers/chat.py needs no such hardcode - it already takes user_id
    # straight from the request, since shopassist-client sends one at login.

    #Interaction 1: Order Status
    current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    customer_query_1 = CustomerQuery(session_id=current_session_id, user_id="alum-1001", text="Hi, I'd like to check my order status for order ord-1001.")
    print(f"\n>>> Customer: '{customer_query_1.text}' (Session: {customer_query_1.session_id})")
    response_1 = orchestrator.handle_customer_query(customer_query_1)
    print(f"\n<<< Chatbot: '{response_1.response_text}' (Agent: {response_1.agent_invoked})")
    print("-" * 80)

    # Interaction 2: Order Deletion
    #current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    #customer_query_2 = CustomerQuery(session_id=current_session_id, user_id="alum-1003", text="Hi, I dont need this order ord-1003.")
    #print(f"\n>>> Customer: '{customer_query_2.text}' (Session: {customer_query_2.session_id})")
    #response_2 = orchestrator.handle_customer_query(customer_query_2)
    #print(f"\n<<< Chatbot: '{response_2.response_text}' (Agent: {response_2.agent_invoked})")
    #print("-" * 80)

    # Interaction 2: Product Recommendation
    #current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    #customer_query_2 = CustomerQuery(session_id=current_session_id, user_id="alum-1001", text="Can you recommend a good laptop for gaming?")
    #print(f"\n>>> Customer: '{customer_query_2.text}' (Session: {customer_query_2.session_id})")
    #response_2 = orchestrator.handle_customer_query(customer_query_2)
    #print(f"\n<<< Chatbot: '{response_2.response_text}' (Agent: {response_2.agent_invoked})")
    #print("-" * 80)

    # Interaction 3: General Query with PII (should be masked)
    #current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    #customer_query_3 = CustomerQuery(session_id=current_session_id, user_id="alum-1001", text="What's your return policy? My email is John.Doe@example.com.")
    #print(f"\n>>> Customer: '{customer_query_3.text}' (Session: {customer_query_3.session_id})")
    #response_3 = orchestrator.handle_customer_query(customer_query_3)
    #print(f"\n<<< Chatbot: '{response_3.response_text}' (Agent: {response_3.agent_invoked})")
    #print("-" * 80)

    # Interaction 4: Order Status with PII (should be masked & new order)
    # Using existing session to show history awareness (though simple in mock)
    #customer_query_4 = CustomerQuery(session_id=customer_query_1.session_id, user_id="alum-1001", text="Actually, my name is Jane Smith. What about order 54321, is that shipped?")
    #print(f"\n>>> Customer: '{customer_query_4.text}' (Session: {customer_query_4.session_id})")
    #response_4 = orchestrator.handle_customer_query(customer_query_4)
    #print(f"\n<<< Chatbot: '{response_4.response_text}' (Agent: {response_4.agent_invoked})")
    #print("-" * 80)

    # Interaction 5: Query leading to GeneralPurpose Agent
    #current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    #customer_query_5 = CustomerQuery(session_id=current_session_id, user_id="alum-1001", text="Tell me about your company's history.")
    #print(f"\n>>> Customer: '{customer_query_5.text}' (Session: {customer_query_5.session_id})")
    #response_5 = orchestrator.handle_customer_query(customer_query_5)
    #print(f"\n<<< Chatbot: '{response_5.response_text}' (Agent: {response_5.agent_invoked})")
    #print("-" * 80)

    print("\n--- Simulation Complete ---")

     # --- Langfuse Integration Start: Flush Traces ---
    # For a short-lived script like main_simulation, an explicit flush
    # ensures all traces are sent before the program exits, providing immediate feedback.
    print("--- Flushing Langfuse traces ---")
    langfuse_client = get_langfuse_client_instance()
    if langfuse_client:
        langfuse_client.flush()
        # Removed: langfuse_client.wait_for_flush()
    print("--- Langfuse traces flushed ---")
    # --- Langfuse Integration End ---