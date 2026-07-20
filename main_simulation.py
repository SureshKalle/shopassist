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
# --- MODIFIED: Removed imports for ReportLab, kept shutil for initial cleanup ---
import shutil # For removing the docs folder and its contents
# END NEW IMPORTS

from db.init_db import build_db
from common.models import CustomerQuery, RawCustomerConversation, RawProductRecord
from clients.ecommerce_api_client import EcommerceClient
from services.pii_masker import PIIMasker
from services.llm_inference import LLMInferenceService
from services.classifier_client import ClassifierClient
# --- MODIFIED: Import RAGService, not MockRAGService ---
from services.rag import RAGService
# --- END MODIFIED ---
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
    #build_db()
    
    # 1. Initialize Core Services
    pii_masker = PIIMasker()
    llm_inference_service = LLMInferenceService()

    # Define default file paths as per RAGService's __init__
    DEFAULT_RAG_DATA_FILE = "rag_knowledge_base.jsonl"
    DEFAULT_FAISS_INDEX_FILE = "faiss_index.bin"
    DEFAULT_DOCS_FOLDER = "docs" # Standard folder for user-provided documents

    # --- MODIFIED: Initial Cleanup Block (can be commented out to load existing files) ---
    # print("\n--- Cleaning up previous default RAG and docs data ---")
    # if os.path.exists(DEFAULT_RAG_DATA_FILE):
    #     os.remove(DEFAULT_RAG_DATA_FILE)
    #     print(f"Removed existing RAG data file: {DEFAULT_RAG_DATA_FILE}")
    # if os.path.exists(DEFAULT_FAISS_INDEX_FILE):
    #     os.remove(DEFAULT_FAISS_INDEX_FILE)
    #     print(f"Removed existing FAISS index file: {DEFAULT_FAISS_INDEX_FILE}")
    # if os.path.exists(DEFAULT_DOCS_FOLDER):
    #     shutil.rmtree(DEFAULT_DOCS_FOLDER) # Removes folder and its contents
    #     print(f"Removed existing docs folder: {DEFAULT_DOCS_FOLDER}")
    # # Recreate the docs folder so ingest_pdf_documents has a target, even if empty initially
    # os.makedirs(DEFAULT_DOCS_FOLDER, exist_ok=True)
    # print(f"Ensured '{DEFAULT_DOCS_FOLDER}' directory exists for user-provided PDFs.")
    # print("--- Initial Cleanup complete ---")
    # --- END MODIFIED ---

    # Initialize RAGService without explicit paths to use its defaults
    rag_service = RAGService(llm_inference_service)
    print(f"Initialized RAGService. Using default JSONL: {DEFAULT_RAG_DATA_FILE}, default FAISS index: {DEFAULT_FAISS_INDEX_FILE}")
    
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
    # --- MODIFIED: Use doc_store for size ---
    print(f"Product RAG doc_store now contains {len(rag_service.doc_store)} documents from initial ingestion.")
    # --- END MODIFIED ---

    # --- MODIFIED: Ingest PDF documents (from user-provided 'docs' folder) ---
    # Check if there are any PDF files in the default docs folder
    pdf_files_in_docs_folder = [f for f in os.listdir(DEFAULT_DOCS_FOLDER) if f.lower().endswith(".pdf")]
    
    if pdf_files_in_docs_folder:
        print(f"\n--- Ingesting PDF Documents from '{DEFAULT_DOCS_FOLDER}' for RAG Knowledge Base ---")
        # --- NEW: dummy PDF creation for testing purposes if no real PDFs are found yet.
        # This block is *only* to ensure a PDF exists for the RAG test to run successfully if
        # the user hasn't placed one. This makes the simulation more robust.
        # If you always place your own PDFs, you can comment this specific inner block out.
        # This requires 'reportlab'.
        if not any(f.lower().endswith(".pdf") for f in os.listdir(DEFAULT_DOCS_FOLDER)):
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
                dummy_pdf_path_for_rag_test = os.path.join(DEFAULT_DOCS_FOLDER, "sim_temp_policy.pdf")
                c = canvas.Canvas(dummy_pdf_path_for_rag_test, pagesize=letter)
                c.drawString(100, 750, "Simulated Return Policy for Electronics")
                c.drawString(100, 720, "Electronics can be returned within 15 days. Warranty claims valid for 1 year.")
                c.drawString(100, 690, "For support, email us at contact@[EMAIL]. My name is [NAME].")
                c.save()
                print(f"Temporarily created dummy PDF for RAG test: {dummy_pdf_path_for_rag_test}")
                pdf_files_in_docs_folder.append(os.path.basename(dummy_pdf_path_for_rag_test)) # Ensure it's processed
            except ImportError:
                print("Warning: 'reportlab' not installed. Cannot create temporary PDF for RAG test if 'docs' folder is empty.")
            except Exception as e:
                print(f"Error creating temporary PDF for RAG test: {e}")
        # --- End of temporary PDF creation ---

        num_pdf_chunks = data_pipeline.ingest_pdf_documents(docs_folder=DEFAULT_DOCS_FOLDER, source_type="customer_policy")
        print(f"Ingested {num_pdf_chunks} chunks from PDF documents into RAG.")
        print(f"Total RAG doc_store size after PDF ingestion: {len(rag_service.doc_store)} documents.")
    else:
        print(f"\n--- Skipping PDF ingestion: No PDF files found in '{DEFAULT_DOCS_FOLDER}'. Please add PDFs there to enable this. ---") 
    # --- END MODIFIED --- 

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
    #current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    #customer_query_1 = CustomerQuery(session_id=current_session_id, user_id="alum-1001", text="Hi, I'd like to check my order status for order ord-1001.")
    #print(f"\n>>> Customer: '{customer_query_1.text}' (Session: {customer_query_1.session_id})")
    #response_1 = orchestrator.handle_customer_query(customer_query_1)
    #print(f"\n<<< Chatbot: '{response_1.response_text}' (Agent: {response_1.agent_invoked})")
    #print("-" * 80)

    # --- NEW ACTIVE INTERACTION: Query about RAG-ingested policy (unconditional) ---
    current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    # This query is designed to hit the policy info from the ingested data (product catalog, customer conversations, OR PDFs)
    customer_query_rag_test = CustomerQuery(session_id=current_session_id, user_id="alum-1001", text="What is your domestic shipping policy and Where is my order ord-1001?")
    print(f"\n>>> Customer (RAG Test): '{customer_query_rag_test.text}' (Session: {customer_query_rag_test.session_id})")
    response_rag_test = orchestrator.handle_customer_query(customer_query_rag_test)
    print(f"\n<<< Chatbot (RAG Test): '{response_rag_test.response_text}' (Agent: {response_rag_test.agent_invoked})")
    print("-" * 80)
    # --- END NEW ACTIVE INTERACTION ---

    # Interaction 2: Order Deletion (COMMENTED OUT)
    #current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    #customer_query_2 = CustomerQuery(session_id=current_session_id, user_id="alum-1003", text="Hi, I dont need this order ord-1003.")
    #print(f"\n>>> Customer: '{customer_query_2.text}' (Session: {customer_query_2.session_id})")
    #response_2 = orchestrator.handle_customer_query(customer_query_2)
    #print(f"\n<<< Chatbot: '{response_2.response_text}' (Agent: {response_2.agent_invoked})")
    #print("-" * 80)

    # Interaction 2: Product Recommendation (COMMENTED OUT)
    #current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    #customer_query_2 = CustomerQuery(session_id=current_session_id, user_id="alum-1001", text="Can you recommend a good laptop for gaming?")
    #print(f"\n>>> Customer: '{customer_query_2.text}' (Session: {customer_query_2.session_id})")
    #response_2 = orchestrator.handle_customer_query(customer_query_2)
    #print(f"\n<<< Chatbot: '{response_2.response_text}' (Agent: {response_2.agent_invoked})")
    #print("-" * 80)

    # Interaction 3: General Query with PII (should be masked) (COMMENTED OUT)
    #current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    #customer_query_3 = CustomerQuery(session_id=current_session_id, user_id="alum-1001", text="What's your return policy? My email is John.Doe@example.com.")
    #print(f"\n>>> Customer: '{customer_query_3.text}' (Session: {customer_query_3.session_id})")
    #response_3 = orchestrator.handle_customer_query(customer_query_3)
    #print(f"\n<<< Chatbot: '{response_3.response_text}' (Agent: {response_3.agent_invoked})")
    #print("-" * 80)

    # Interaction 4: Order Status with PII (should be masked & new order) (COMMENTED OUT)
    # Using existing session to show history awareness (though simple in mock)
    #customer_query_4 = CustomerQuery(session_id=customer_query_1.session_id, user_id="alum-1001", text="Actually, my name is Jane Smith. What about order 54321, is that shipped?")
    #print(f"\n>>> Customer: '{customer_query_4.text}' (Session: {customer_query_4.session_id})")
    #response_4 = orchestrator.handle_customer_query(customer_query_4)
    #print(f"\n<<< Chatbot: '{response_4.response_text}' (Agent: {response_4.agent_invoked})")
    #print("-" * 80)

    # Interaction 5: Query leading to GeneralPurpose Agent (COMMENTED OUT)
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

