import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

# --- Customer Interaction Models ---
class CustomerQuery(BaseModel):
    session_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    text: str
    source_channel: str = "web_chat" # e.g., web_chat, twitter, mobile_app

class MaskedQuery(BaseModel):
    session_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    masked_text: str
    original_text_hash: str # To reference original for audit, but not store PII

class ChatbotResponse(BaseModel):
    session_id: str
    response_text: str
    agent_invoked: Optional[str] = None
    confidence_score: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.now)

# --- LLM Inference Service Request/Response Models ---
class RoutingRequest(BaseModel):
    session_id: str
    conversation_history: List[Dict[str, str]] # PII-masked history
    current_query: str # PII-masked query

class AgentInvocation(BaseModel):
    agent_name: str # e.g., "OrderTrackingAgent", "ProductRecommendationAgent"
    confidence: float
    parameters: Dict[str, Any] # Structured parameters for the agent

class AgentTask(BaseModel):
    session_id: str
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str # For internal agent lookup, could be masked/tokenized
    original_query: str # Masked
    intent: str
    params: Dict[str, Any] # e.g., {'order_id': '123'}
    conversation_context: List[Dict[str, str]] # Masked

class StructuredOrderSummary(BaseModel):
    order_id: str
    status: str
    items: List[Dict[str, Any]]
    estimated_delivery: Optional[str]
    issue_analysis: Optional[str]

class StructuredProductRecommendation(BaseModel):
    product_id: str
    name: str
    description_snippet: str
    price: float
    reason: str

# Union type for common structured results from agents
AgentResultData = Union[StructuredOrderSummary, StructuredProductRecommendation, Dict[str, Any]]

class StructuredAgentResult(BaseModel):
    task_id: str
    agent_name: str
    status: str # 'success', 'failure', 'escalation'
    result_data: AgentResultData
    # For a real system, this might be a discriminated union or a more complex generic.

class LLMAgentReasonRequest(BaseModel):
    session_id: str
    agent_name: str
    task_description: str
    current_state: Dict[str, Any]
    available_tools: List[str] # e.g., ['ECommerceAPI.getOrderDetails', 'RAG.queryPolicy']

class LLMAgentReasonResponse(BaseModel):
    action: str # e.g., 'call_api', 'query_rag', 'return_result', 'escalate'
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    thought: str

class LLMAgentInterpretRequest(BaseModel):
    session_id: str
    agent_name: str
    raw_data: Dict[str, Any] # e.g., raw API response for order details
    interpretation_goal: str # e.g., 'diagnose order issue', 'summarize product features'

class LLMAgentInterpretResponse(BaseModel):
    structured_interpretation: Any # e.g., OrderIssueAnalysis schema
    thought: str

class NLGRequest(BaseModel):
    session_id: str
    conversation_history: List[Dict[str, str]] # PII-masked
    agent_results: List[StructuredAgentResult]
    final_user_intent: str # As interpreted by Orchestrator
    customer_name: Optional[str] = None # Looked up by customer_id; None if unknown/unresolvable

# --- Data Ingestion Models ---
class RawCustomerConversation(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] # e.g., source_platform, user_handle

class CleanedCustomerConversation(BaseModel):
    id: str
    cleaned_text: str
    tokens: List[str]
    metadata: Dict[str, Any]

class RawProductRecord(BaseModel):
    product_id: str
    raw_description: str
    specs: Dict[str, Any]
    reviews: List[str]
    price: str

class CleanedProductRecord(BaseModel):
    product_id: str
    clean_description: str
    structured_specs: Dict[str, Any]
    sentiment_analyzed_reviews: List[Dict[str, Any]]
    normalized_price: float
    metadata: Dict[str, Any]

class ChunkedDocument(BaseModel):
    doc_id: str
    content: str
    embedding: List[float]
    source_type: str # 'product_catalog', 'customer_support_policy'
    metadata: Dict[str, Any]
