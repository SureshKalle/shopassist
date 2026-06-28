# main_simulation.py
import uuid
from datetime import datetime

# Import all services and models
from common.models import CustomerQuery, RawCustomerConversation, RawProductRecord
from clients.ecommerce_api_client import MockECommerceAPIClient
from services.pii_masker import PIIMasker
from services.llm_inference import MockLLMInferenceService
from services.rag import MockRAGService
from services.data_pipeline import DataIngestionPipeline
from services.agents.order_tracking_agent import OrderTrackingAgent
from services.agents.product_recommendation_agent import ProductRecommendationAgent
from services.agents.general_purpose_agent import GeneralPurposeAgent
from services.agents.escalation_agent import EscalationAgent
from services.orchestrator import AgentOrchestratorService

# --- Main Execution Flow (Simulates System Startup and Customer Interactions) ---

if __name__ == "__main__":
    print("--- Initializing Chatbot System Components ---")

    # 1. Initialize Core Services
    pii_masker = PIIMasker()
    llm_inference_service = MockLLMInferenceService()
    rag_service = MockRAGService(llm_inference_service) # RAG needs LLM for embeddings
    ecommerce_api_client = MockECommerceAPIClient()

    # 2. Initialize Data Ingestion Pipeline
    data_pipeline = DataIngestionPipeline(pii_masker, llm_inference_service, rag_service)

    # --- SIMULATE DATA PREPARATION & INGESTION ---
    print("\n--- Running Data Preparation & Ingestion Cycle ---")
    
    # Raw Customer Conversations
    raw_customer_conversations = [
        RawCustomerConversation(id="conv_001", text="Hi, my name is John Doe, and I want to know about my order 12345.", metadata={"source": "twitter", "user_id": "jd_123"}),
        RawCustomerConversation(id="conv_002", text="Can you help me with a return for product X? My email is john.doe@example.com.", metadata={"source": "web_form", "user_id": "jd_123"}),
        RawCustomerConversation(id="conv_003", text="I love my new laptop! Is there a warranty?", metadata={"source": "web_chat", "user_id": "cust_002"}),
    ]
    cleaned_convs = data_pipeline.ingest_customer_conversations(raw_customer_conversations)
    print(f"\nSample Cleaned Conversation for LLM Fine-tuning: '{cleaned_convs[0].cleaned_text[:50]}...'")

    # Raw E-commerce Product Catalog
    raw_product_catalog = [
        RawProductRecord(product_id="PROD_LAP_001", raw_description="High-performance gaming laptop with an i7 processor, 16GB RAM, and a 1TB SSD. Stunning display and RGB keyboard.", specs={"CPU": "i7", "RAM": "16GB", "Storage": "1TB SSD"}, reviews=["Great product!", "Fast delivery.", "Screen is amazing!"], price="$1200.00"),
        RawProductRecord(product_id="PROD_HEAD_002", raw_description="Premium noise-cancelling headphones for immersive audio. Comfortable earcups and 20-hour battery life.", specs={"Color": "Black", "Battery": "20h"}, reviews=["Awesome sound!", "John Doe found them comfy and fit perfectly."], price="$250.00"),
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
    orchestrator = AgentOrchestratorService(llm_inference_service, pii_masker, agents)
    print("Initialized Agent Orchestrator Service.")

    print("\n--- SYSTEM READY: Simulating Customer Interactions ---\n")

    # --- SIMULATE CUSTOMER INTERACTIONS ---

    # Interaction 1: Order Status
    current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    customer_query_1 = CustomerQuery(session_id=current_session_id, user_id="cust_001", text="Hi, I'd like to check my order status for order 12345.")
    print(f"\n>>> Customer: '{customer_query_1.text}' (Session: {customer_query_1.session_id})")
    response_1 = orchestrator.handle_customer_query(customer_query_1)
    print(f"\n<<< Chatbot: '{response_1.response_text}' (Agent: {response_1.agent_invoked})")
    print("-" * 80)

    # Interaction 2: Product Recommendation
    current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    customer_query_2 = CustomerQuery(session_id=current_session_id, user_id="cust_002", text="Can you recommend a good laptop for gaming?")
    print(f"\n>>> Customer: '{customer_query_2.text}' (Session: {customer_query_2.session_id})")
    response_2 = orchestrator.handle_customer_query(customer_query_2)
    print(f"\n<<< Chatbot: '{response_2.response_text}' (Agent: {response_2.agent_invoked})")
    print("-" * 80)

    # Interaction 3: General Query with PII (should be masked)
    current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    customer_query_3 = CustomerQuery(session_id=current_session_id, user_id="cust_003", text="What's your return policy? My email is John.Doe@example.com.")
    print(f"\n>>> Customer: '{customer_query_3.text}' (Session: {customer_query_3.session_id})")
    response_3 = orchestrator.handle_customer_query(customer_query_3)
    print(f"\n<<< Chatbot: '{response_3.response_text}' (Agent: {response_3.agent_invoked})")
    print("-" * 80)

    # Interaction 4: Order Status with PII (should be masked & new order)
    # Using existing session to show history awareness (though simple in mock)
    customer_query_4 = CustomerQuery(session_id=customer_query_1.session_id, user_id="cust_001", text="Actually, my name is Jane Smith. What about order 54321, is that shipped?")
    print(f"\n>>> Customer: '{customer_query_4.text}' (Session: {customer_query_4.session_id})")
    response_4 = orchestrator.handle_customer_query(customer_query_4)
    print(f"\n<<< Chatbot: '{response_4.response_text}' (Agent: {response_4.agent_invoked})")
    print("-" * 80)

    # Interaction 5: Query leading to GeneralPurpose Agent
    current_session_id = f"user_session_{uuid.uuid4().hex[:8]}"
    customer_query_5 = CustomerQuery(session_id=current_session_id, user_id="cust_004", text="Tell me about your company's history.")
    print(f"\n>>> Customer: '{customer_query_5.text}' (Session: {customer_query_5.session_id})")
    response_5 = orchestrator.handle_customer_query(customer_query_5)
    print(f"\n<<< Chatbot: '{response_5.response_text}' (Agent: {response_5.agent_invoked})")
    print("-" * 80)

    print("\n--- Simulation Complete ---")