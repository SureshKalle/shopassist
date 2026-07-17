# api/routers/chat.py
"""
Chat endpoints — the primary customer-facing conversational API.

POST   /api/v1/chat                      → send a message, get a response
GET    /api/v1/chat/{session_id}/history → retrieve conversation history
DELETE /api/v1/chat/{session_id}         → clear a session
"""

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException

from api.config import settings
from api.dependencies import get_orchestrator
from api.rate_limit import rate_limit
from api.schemas import ChatRequest, ChatResponse
from api.security import verify_api_key
from common.models import CustomerQuery
from services.orchestrator import AgentOrchestratorService

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit)],
)


@router.post("", response_model=ChatResponse)
async def send_message(
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
        # orchestrator.handle_customer_query is blocking (sync LLM/DB calls),
        # so it runs in a worker thread; wait_for bounds how long *this
        # request* waits on it. It does not cancel the thread itself - Python
        # can't force-cancel a running thread - so a timeout here frees up
        # the client and this request, but the orchestrator call keeps
        # running in the background until it finishes on its own. Real
        # cancellation needs async LLM/DB clients throughout services/.
        response = await asyncio.wait_for(
            asyncio.to_thread(orchestrator.handle_customer_query, query),
            timeout=settings.chat_request_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        logger.error(
            "[API] Orchestrator timed out after %ss (session=%s)",
            settings.chat_request_timeout_seconds, session_id,
        )
        raise HTTPException(status_code=504, detail="Request timed out while processing your message.") from exc
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
