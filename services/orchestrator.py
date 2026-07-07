# services/orchestrator.py
"""
AgentOrchestratorService — the central brain of the multi-agent system.
LangChain edition: conversation memory now uses `InMemoryChatMessageHistory`
(LangChain's standard memory abstraction) instead of a raw dict, so it's a
drop-in swap to `RedisChatMessageHistory` or any other LangChain-community
history backend for production persistence.

Responsibilities:
  1. PII masking of incoming customer text
  2. Conversation history management (per-session, via LangChain message history)
  3. Intent routing → LLMInferenceService.call_router (with Cache1)
  4. Task delegation to the correct specialised LangChain agent
  5. Error/escalation fallback to EscalationAgent
  6. Final natural-language response generation (NLG)
  7. Conversation history update & agent state persistence

Cache1 (orchestrator_routing_cache):
  Keyed by masked query text. In production, swap the in-process dict for
  Redis (TTL ~5 min) — the interface is identical.
"""

import logging
from typing import Optional

from langchain_core.chat_history import InMemoryChatMessageHistory

from clients.ecommerce_api_client import MockECommerceAPIClient
from common.models import (
    AgentInvocation,
    AgentTask,
    ChatbotResponse,
    CustomerQuery,
    NLGRequest,
    RoutingRequest,
    StructuredAgentResult,
)
from services.agents.base_agent import BaseAgent, EscalationSignal
from services.llm_inference import LLMInferenceService
from services.pii_masker import PIIMasker

logger = logging.getLogger(__name__)


class AgentOrchestratorService:
    """
    Central orchestrator for the AI Agentic Customer Support Platform.

    Args:
        llm_inference_client : LLMInferenceService instance.
        pii_masker           : PIIMasker instance.
        agents               : Dict mapping agent_name → BaseAgent instance.
                               Must include "EscalationAgent".
    """

    def __init__(
        self,
        llm_inference_client: LLMInferenceService,
        pii_masker: PIIMasker,
        agents: dict[str, BaseAgent],
    ):
        self.llm_inference_client = llm_inference_client
        self.pii_masker = pii_masker
        self.agents = agents

        # ── Conversation memory (LangChain standard interface) ────────────
        # {session_id: InMemoryChatMessageHistory}
        # Swap for RedisChatMessageHistory(session_id, url=...) in production —
        # same .add_user_message / .add_ai_message / .messages interface.
        self._histories: dict[str, InMemoryChatMessageHistory] = {}

        # {session_id: {"last_agent": str, "last_result": List[StructuredAgentResult]}}
        self.agent_state_store: dict[str, dict] = {}

        # Cache1: {masked_query_text: AgentInvocation}
        self.orchestrator_routing_cache: dict[str, AgentInvocation] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Memory helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._histories:
            self._histories[session_id] = InMemoryChatMessageHistory()
        return self._histories[session_id]

    @staticmethod
    def _history_as_dicts(history: InMemoryChatMessageHistory) -> list[dict[str, str]]:
        """Convert LangChain BaseMessage objects to the {"role", "content"} dicts
        the rest of the pipeline (RoutingRequest, NLGRequest) expects."""
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        return [
            {
                "role": role_map.get(m.type, m.type),
                "content": m.content if isinstance(m.content, str) else str(m.content),
            }
            for m in history.messages
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    def handle_customer_query(self, query: CustomerQuery) -> ChatbotResponse:
        """Process a single customer query through the full agent pipeline."""
        session_id = query.session_id
        user_id = query.user_id
        logger.info("[Orchestrator] Handling query | session=%s user=%s", session_id, user_id)

        # ── Step 1: PII Masking ────────────────────────────────────────
        masked_query = self.pii_masker.mask_text(query.text, session_id=session_id, user_id=user_id)
        logger.info("[Orchestrator] Masked query: '%s'", masked_query.masked_text)

        # ── Step 2: Conversation history (LangChain memory) ─────────────
        history = self._get_history(session_id)
        history.add_user_message(masked_query.masked_text)
        current_history = self._history_as_dicts(history)

        # ── Step 3: Intent routing (with Cache1) ───────────────────────
        agent_invocation = self._route_query(session_id, masked_query.masked_text, current_history)
        logger.info(
            "[Orchestrator] Routing → agent='%s' confidence=%.2f",
            agent_invocation.agent_name, agent_invocation.confidence,
        )

        # ── Step 4: Agent task delegation ─────────────────────────────
        agent_results = self._delegate_to_agent(
            agent_invocation, session_id, user_id, masked_query.masked_text, current_history
        )

        # ── Step 5: Natural Language Generation ───────────────────────
        nlg_request = NLGRequest(
            session_id=session_id,
            conversation_history=current_history,
            agent_results=agent_results,
            final_user_intent=agent_invocation.agent_name,
        )
        final_response_text = self.llm_inference_client.call_generative(nlg_request)
        logger.info("[Orchestrator] NLG response generated (%d chars).", len(final_response_text))

        # ── Step 6: Update memory & state ──────────────────────────────
        history.add_ai_message(final_response_text)
        self.agent_state_store[session_id] = {
            "last_agent": agent_invocation.agent_name,
            "last_result": agent_results,
        }

        confidence = agent_invocation.confidence if agent_results else 0.3
        if agent_results and agent_results[0].status == "escalation":
            confidence = 0.0

        return ChatbotResponse(
            session_id=session_id,
            response_text=final_response_text,
            agent_invoked=agent_invocation.agent_name,
            confidence_score=confidence,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Routing
    # ──────────────────────────────────────────────────────────────────────────

    def _route_query(
        self, session_id: str, masked_query_text: str, current_history: list[dict[str, str]]
    ) -> AgentInvocation:
        """Route the query, using Cache1 for repeated identical messages."""
        cache_key = masked_query_text
        if cache_key in self.orchestrator_routing_cache:
            logger.debug("[Orchestrator] Cache1 hit for routing.")
            return self.orchestrator_routing_cache[cache_key]

        routing_request = RoutingRequest(
            session_id=session_id,
            conversation_history=current_history,
            current_query=masked_query_text,
        )
        invocation = self.llm_inference_client.call_router(routing_request)
        self.orchestrator_routing_cache[cache_key] = invocation
        return invocation

    # ──────────────────────────────────────────────────────────────────────────
    # Task delegation
    # ──────────────────────────────────────────────────────────────────────────

    def _delegate_to_agent(
        self,
        agent_invocation: AgentInvocation,
        session_id: str,
        user_id: str,
        masked_query_text: str,
        current_history: list[dict[str, str]],
    ) -> list[StructuredAgentResult]:
        """Build an AgentTask, dispatch to the correct agent, and handle failures."""
        target_agent: Optional[BaseAgent] = self.agents.get(agent_invocation.agent_name)
        if not target_agent:
            logger.warning(
                "[Orchestrator] Agent '%s' not registered — escalating.", agent_invocation.agent_name
            )
            original_intent = agent_invocation.agent_name
            agent_invocation.agent_name = "EscalationAgent"
            agent_invocation.parameters = {
                "reason": f"No agent registered for intent: {original_intent}"
            }
            target_agent = self.agents["EscalationAgent"]

        agent_task = AgentTask(
            session_id=session_id,
            customer_id=user_id,
            original_query=masked_query_text,
            intent=agent_invocation.agent_name,
            params=agent_invocation.parameters,
            conversation_context=current_history,
        )

        results: list[StructuredAgentResult] = []
        try:
            results.append(target_agent.process_task(agent_task))

        except EscalationSignal as esc:
            logger.warning("[Orchestrator] EscalationSignal from %s: %s", target_agent.name, esc.reason)
            escalation_task = AgentTask(
                session_id=session_id, customer_id=user_id,
                original_query=masked_query_text, intent="escalation_signal",
                params={"reason": esc.reason}, conversation_context=current_history,
            )
            results.append(self.agents["EscalationAgent"].process_task(escalation_task))

        except Exception as exc:
            logger.error(
                "[Orchestrator] Unhandled error in agent %s: %s", target_agent.name, exc, exc_info=True
            )
            escalation_task = AgentTask(
                session_id=session_id, customer_id=user_id,
                original_query=masked_query_text, intent="agent_error",
                params={"reason": f"Agent {target_agent.name} failed: {exc}"},
                conversation_context=current_history,
            )
            results.append(self.agents["EscalationAgent"].process_task(escalation_task))

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        """Return the full conversation history for a session."""
        if session_id not in self._histories:
            return []
        return self._history_as_dicts(self._histories[session_id])

    def clear_session(self, session_id: str) -> None:
        """Clear conversation state for a session (e.g. on logout)."""
        self._histories.pop(session_id, None)
        self.agent_state_store.pop(session_id, None)
        logger.info("[Orchestrator] Cleared session %s.", session_id)
