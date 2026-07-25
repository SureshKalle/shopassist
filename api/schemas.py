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
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Existing session ID to continue a conversation. Omit to start a new session."
    )
    user_id: str = Field(description="Identifier for the customer sending the message.")
    text: str = Field(description="The customer's message text.", min_length=1, max_length=2000)
    source_channel: str = Field(default="web_chat", description="Origin channel, e.g. web_chat, mobile_app, twitter.")


class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    agent_invoked: Optional[str] = None
    confidence_score: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    registered_agents: list[str]
    rag_documents_indexed: int
    database_reachable: bool
    llm_reachable: bool
    classifier_reachable: bool
    api_key_enforced: bool

