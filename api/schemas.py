# api/schemas.py
"""
FastAPI request/response schemas.

Kept separate from `common/models.py` deliberately: `common/models.py`
defines the *internal* domain models shared across services/agents, while
these schemas define the *external* HTTP contract. Keeping them distinct
means internal refactors (e.g. renaming an internal field) don't
automatically break the public API contract, and vice versa — the API
layer can evolve its request/response shape independently.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Existing session ID to continue a conversation. Omit to start a new session."
    )
    user_id: str = Field(description="Identifier for the customer sending the message.")
    text: str = Field(description="The customer's message text.", min_length=1)
    source_channel: str = Field(default="web_chat", description="Origin channel, e.g. web_chat, mobile_app, twitter.")


class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    agent_invoked: Optional[str] = None
    confidence_score: float
    timestamp: datetime


class HistoryTurn(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: list[HistoryTurn]


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------
class PolicyDocumentIn(BaseModel):
    id: str
    title: str
    content: str


class IngestPoliciesRequest(BaseModel):
    policies: list[PolicyDocumentIn]


class ProductRecordIn(BaseModel):
    product_id: str
    raw_description: str
    specs: dict[str, Any] = Field(default_factory=dict)
    reviews: list[str] = Field(default_factory=list)
    price: str


class IngestProductsRequest(BaseModel):
    products: list[ProductRecordIn]


class IngestResponse(BaseModel):
    ingested_count: int
    rag_collection_size: int
    message: str


# ---------------------------------------------------------------------------
# Evaluation endpoint
# ---------------------------------------------------------------------------
class EvaluationRequest(BaseModel):
    questions: list[str]
    answers: list[str]
    contexts: list[list[str]]
    ground_truths: list[str]
    latencies_seconds: list[float] = Field(default_factory=list)
    task_statuses: list[str] = Field(default_factory=list)


class MetricResultOut(BaseModel):
    name: str
    value: Optional[float]
    target: float
    comparator: str
    passed: Optional[bool]
    note: str = ""


class EvaluationResponse(BaseModel):
    metrics: list[MetricResultOut]
    overall_pass: bool


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_available: bool
    embedding_provider: str
    vector_store: str
    rag_collection_size: int
    registered_agents: list[str]
