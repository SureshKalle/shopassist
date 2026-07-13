# api/routers/chat.py
"""
Chat endpoints — the primary customer-facing conversational API.

POST /api/v1/chat → send a message, get the orchestrated response.

There's no session history/clear endpoint here — AgentOrchestratorService
doesn't expose one (its history dict is internal), so those routes were
removed rather than shipped calling a method that doesn't exist.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_orchestrator
from api.schemas import ChatRequest, ChatResponse
from common.models import CustomerQuery
from services.orchestrator import AgentOrchestratorService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def send_message(
    request: ChatRequest,
    orchestrator: AgentOrchestratorService = Depends(get_orchestrator),
) -> ChatResponse:
    """
    Send a customer message and receive the orchestrated multi-agent response.

    If `session_id` is omitted, a new session is created and returned in the
    response — the client should persist and reuse it for subsequent turns
    in the same conversation.
    """
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"

    query = CustomerQuery(
        session_id=session_id,
        user_id=request.user_id,
        text=request.text,
        source_channel=request.source_channel,
    )

    try:
        response = orchestrator.handle_customer_query(query)
    except Exception as exc:
        logger.error("[API] Unhandled error in orchestrator: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error processing chat message.") from exc

    return ChatResponse(
        session_id=response.session_id,
        response_text=response.response_text,
        agent_invoked=response.agent_invoked,
        confidence_score=response.confidence_score,
        timestamp=response.timestamp,
    )
