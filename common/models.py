# common/models.py
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

# --- CORE UTILITY MODELS ---
# 1. Message: Used for clear conversation history structure
class Message(BaseModel):
    role: str # e.g., "user", "assistant"
    content: str

# --- AGENT INPUT PARAMETER MODELS ---
# These must be defined before AgentInputParams Union and before AgentInvocation/AgentTask
class OrderTrackingAgentInputParams(BaseModel):
    order_id: str
    customer_id: str # Pseudonymized ID
    # Add other parameters specific to order tracking, e.g., 'date_range', 'item_name'

class ProductRecommendationAgentInputParams(BaseModel):
    customer_id: str # Pseudonymized ID
    product_category_preference: Optional[str] = None
    specific_product_keywords: Optional[str] = None
    # Add context like 'current_page', 'previous_viewed_product_ids'

class GeneralPurposeAgentInputParams(BaseModel):
    query: str # The PII-masked user query
    # If the GeneralPurposeAgent has any specific structured inputs it expects, add them here.

class EscalationAgentInputParams(BaseModel):
    reason: str
    original_query: Optional[str] = None
    conversation_summary_snippet: Optional[str] = None # A brief text summary of context for human
    # Add context like 'failed_agent_name', 'failure_details'

# --- UNION TYPE FOR ALL AGENT INPUT PARAMETERS ---
AgentInputParams = Union[
    OrderTrackingAgentInputParams,
    ProductRecommendationAgentInputParams,
    GeneralPurposeAgentInputParams,
    EscalationAgentInputParams,
    Dict[str, Any] # Fallback for extremely dynamic cases or generic agents not yet covered
]

# --- LLM INTERPRETATION MODELS ---
# OrderIssueAnalysis: Used by LLMAgentInterpretResponse
class OrderIssueAnalysis(BaseModel):
    issue_type: str # e.g., "PaymentPending", "ShippingDelay", "None", "Unknown"
    recommendation: str # Actionable advice or summary
    severity: Optional[str] = None # e.g., "low", "medium", "high"
    additional_notes: Optional[str] = None # Any extra context

# --- LLM GENERATION MODELS (Agent-specific NL snippets) ---
# AgentGenerationOutput: Used by LLMInferenceService.call_agent_generate
class AgentGenerationOutput(BaseModel):
    generated_text: str
    context_used: Optional[List[str]] = None
    confidence: float = 1.0


# --- AGENT RESULT MODELS ---
# These must be defined before StructuredAgentResult where they are used in a Union
class StructuredOrderSummary(BaseModel):
    order_id: str
    status: str
    items: List[Dict[str, Any]] # Could also be a Pydantic model for LineItem
    estimated_delivery: Optional[str]
    issue_analysis: Optional[str] # This string should come from OrderIssueAnalysis or similar

class StructuredProductRecommendation(BaseModel):
    product_id: str
    name: str
    description_snippet: str
    price: float
    reason: str

# GeneralPurposeAnswer: Specific structured result for GeneralPurposeAgent
class GeneralPurposeAnswer(BaseModel):
    answer_snippet: str
    source_documents_summary: Optional[List[str]] = None # Summaries of RAG docs used
    confidence: float = 1.0

# EscalationDetails: Specific structured result for EscalationAgent
class EscalationDetails(BaseModel):
    escalation_reason: str
    original_query: str
    conversation_summary: List[Message] # Uses the new Message model
    ticket_id: Optional[str] = None
    human_agent_queue: Optional[str] = None

# --- MAIN AGENT OUTPUT MODEL ---
# StructuredAgentResult: This is the common wrapper for all agent outputs.
class StructuredAgentResult(BaseModel):
    task_id: str
    agent_name: str
    status: str # 'success', 'failure', 'escalation'
    result_data: Union[ # Union of all possible structured results from agents
        StructuredOrderSummary,
        StructuredProductRecommendation,
        GeneralPurposeAnswer,
        EscalationDetails,
        AgentGenerationOutput,
        Dict[str, Any] # Keep as a very last resort / for transient states
    ]
    # For a real system, this might be a discriminated union or a more complex generic.

# --- FINAL NLG OUTPUT MODEL ---
# FinalNLGOutput: Used by LLMInferenceService.call_generative
class FinalNLGOutput(BaseModel):
    response_text: str
    tone: Optional[str] = "helpful" # e.g., "polite", "empathetic", "neutral"
    is_complete: bool = True # Indicates if the customer's current query has been fully addressed
    confidence: float = 1.0 # Overall confidence of the generated response


# --- CUSTOMER INTERACTION MODELS ---
# CustomerQuery: Initial input from the customer
class CustomerQuery(BaseModel):
    session_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    text: str
    source_channel: str = "web_chat" # e.g., web_chat, twitter, mobile_app

# MaskedQuery: Output from PII Masker
class MaskedQuery(BaseModel):
    session_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    masked_text: str
    original_text_hash: str # To reference original for audit, but not store PII

# ChatbotResponse: Final output from the Orchestrator to the customer-facing interface
class ChatbotResponse(BaseModel):
    session_id: str
    response_text: str
    agent_invoked: Optional[str] = None
    confidence_score: float = 1.0 # This could be derived from FinalNLGOutput.confidence
    timestamp: datetime = Field(default_factory=datetime.now)


# --- LLM INFERENCE SERVICE REQUEST/RESPONSE MODELS ---
# RoutingRequest: Input to LLMInf_Router
class RoutingRequest(BaseModel):
    session_id: str
    conversation_history: List[Message] # Uses Message model
    current_query: str # PII-masked query

# AgentInvocation: Output from LLMInf_Router, used by Orchestrator
class AgentInvocation(BaseModel):
    agent_name: str # e.g., "OrderTrackingAgent", "ProductRecommendationAgent"
    confidence: float
    parameters: AgentInputParams # Uses AgentInputParams Union

# AgentTask: Input from Orchestrator to Specialized Agents
class AgentTask(BaseModel):
    session_id: str
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str # IMPORTANT: This should be a PSEUDONYMIZED customer ID
    original_query: str # Masked
    intent: str
    params: AgentInputParams # Uses AgentInputParams Union
    conversation_context: List[Message] # Uses Message model

# LLMAgentReasonRequest: Input to LLMInf_AgentReason
class LLMAgentReasonRequest(BaseModel):
    session_id: str
    agent_name: str
    task_description: str
    current_state: Dict[str, Any] # This can remain flexible for now, or become a specific AgentState model
    available_tools: List[str] # e.g., ['ECommerceAPI.getOrderDetails', 'RAG.queryPolicy']

# LLMAgentReasonResponse: Output from LLMInf_AgentReason
class LLMAgentReasonResponse(BaseModel):
    action: str # e.g., 'call_api', 'query_rag', 'return_result', 'escalate'
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    thought: str

# LLMAgentInterpretRequest: Input to LLMInf_AgentInterpret
class LLMAgentInterpretRequest(BaseModel):
    session_id: str
    agent_name: str
    raw_data: Dict[str, Any] # e.g., raw API response for order details
    interpretation_goal: str # e.g., 'diagnose order issue', 'summarize product features'

# LLMAgentInterpretResponse: Output from LLMInf_AgentInterpret
class LLMAgentInterpretResponse(BaseModel):
    structured_interpretation: OrderIssueAnalysis # Uses OrderIssueAnalysis model
    thought: str

# NLGRequest: Input to LLMInferenceService.call_generative (for final NLG)
class NLGRequest(BaseModel):
    session_id: str
    conversation_history: List[Message] # Uses Message model
    agent_results: List[StructuredAgentResult] # Uses StructuredAgentResult
    final_user_intent: str # As interpreted by Orchestrator


# --- DATA INGESTION MODELS ---
# RawCustomerConversation: Input for the data pipeline
class RawCustomerConversation(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] # e.g., source_platform, user_handle

# CleanedCustomerConversation: Intermediate output from data pipeline
class CleanedCustomerConversation(BaseModel):
    id: str
    cleaned_text: str
    tokens: List[str]
    metadata: Dict[str, Any]

# RawProductRecord: Input for the data pipeline
class RawProductRecord(BaseModel):
    product_id: str
    raw_description: str
    specs: Dict[str, Any]
    reviews: List[str]
    price: str

# CleanedProductRecord: Intermediate output from data pipeline
class CleanedProductRecord(BaseModel):
    product_id: str
    clean_description: str
    structured_specs: Dict[str, Any]
    sentiment_analyzed_reviews: List[Dict[str, Any]]
    normalized_price: float
    metadata: Dict[str, Any]

# ChunkedDocument: Final output from data pipeline for RAG
class ChunkedDocument(BaseModel):
    doc_id: str
    content: str
    embedding: List[float]
    source_type: str # 'product_catalog', 'customer_support_policy'
    metadata: Dict[str, Any]