# common/models.py
"""
Every Pydantic model shared across shopassist - request/response shapes for
each LLM call (`services/llm_inference.py`), the agents' own input/output
contracts, and the customer-facing query/response pair. Grouped by where each
model sits in the pipeline (see the section headers below); read top to bottom
to follow one customer message end to end.

Nothing here talks to the LLM, the DB, or HTTP directly - these are pure data
shapes, validated by Pydantic on construction. `services/orchestrator.py` is
the best starting point for seeing most of these models used in sequence.
"""
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
    user_id: str # Schema-driven identifier (db/README.md) - matches customers.user_id
    # Add other parameters specific to order tracking, e.g., 'date_range', 'item_name'

class ProductRecommendationAgentInputParams(BaseModel):
    user_id: str # Schema-driven identifier (db/README.md) - matches customers.user_id
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
    user_id: str # The one identifier, end to end: sent by shopassist-client at login,
    # used for order/customer DB lookups (customers.user_id, db/README.md), and threaded
    # into AgentTask.user_id unchanged.
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

# SentimentResult: shopassist's local mirror of the classifier service's
# response shape (its own SentimentResponse schema) - see
# services/classifier_client.py. Not yet consulted by routing/NLG - today
# it's only logged/recorded (services/orchestrator.py's sentiment hook,
# services/data_pipeline.py's review-sentiment call site).
class SentimentResult(BaseModel):
    label: str # "negative" | "neutral" | "positive" | "unknown"
    stars: int # 1-5, from the underlying star-rating model; 0 if unknown
    score: float # confidence in raw_label, 0-1
    raw_label: str # the model's own output before bucketing, e.g. "4 stars"

# ChatbotResponse: Final output from the Orchestrator to the customer-facing interface
class ChatbotResponse(BaseModel):
    session_id: str
    response_text: str
    agent_invoked: Optional[str] = None
    confidence_score: float = 1.0 # This could be derived from FinalNLGOutput.confidence
    timestamp: datetime = Field(default_factory=datetime.now)


# --- LLM INFERENCE SERVICE REQUEST/RESPONSE MODELS ---
# AgentTask: Input from Orchestrator to Specialized Agents
class AgentTask(BaseModel):
    session_id: str
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str # Schema-driven identifier (db/README.md) - matches customers.user_id
    original_query: str # Masked
    # Overloaded: on the normal decomposition path (services/orchestrator.py)
    # this is the target agent's registry name itself (e.g.
    # "OrderTrackingAgent" - see api/dependencies.py::get_agents()'s
    # docstring), used only for logging on that path, since routing already
    # happened by the time an AgentTask is built. On the guardrail-block/
    # sub-task-error escalation paths it's instead a free-form semantic
    # label ("escalation_due_to_guardrail", "escalation_due_to_sub_task_error")
    # - no agent branches on this field's value, so the inconsistency is
    # harmless today, just worth knowing before adding logic that reads it.
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
    # tool_name can now be a category like "ECommerceAPI" or "RAG"
    # The actual method/operation will be in tool_params.method
    tool_name: Optional[str] = None
    # tool_params will now *always* include a 'method' field for 'call_api'/'query_rag'
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
    # Optional and defaulted to None so existing callers/tests that build an
    # NLGRequest without it keep working unchanged. label="unknown" (or None)
    # means call_generative skips the sentiment-calibration prompt entirely -
    # see that method's comment.
    customer_sentiment: Optional[SentimentResult] = None


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

# --- MULTI-INTENT & TASK DECOMPOSITION MODELS (NEW SECTION) ---

class DecomposedSubTask(BaseModel):
    """Represents a single sub-task extracted from a multi-intent query."""
    sub_task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_segment: str # The part of the original query this sub-task addresses
    inferred_agent_name: str # e.g., "OrderTrackingAgent", "GeneralPurposeAgent"
    inferred_parameters: Dict[str, Any] = Field(default_factory=dict) # Parameters for the agent

class DecomposedQuery(BaseModel):
    """Output model for the LLM that decomposes a multi-intent query."""
    primary_intent_summary: str # A summary of the overall user goal
    sub_tasks: List[DecomposedSubTask] # List of individual sub-tasks
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)