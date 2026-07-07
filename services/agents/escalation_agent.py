# services/agents/escalation_agent.py
"""
EscalationAgent — final fallback for issues that automated agents cannot resolve.

This agent deliberately does NOT use a LangChain tool-calling loop: escalation
is a deterministic bookkeeping operation (build a ticket payload for a human),
not a reasoning task. Keeping it simple avoids burning an LLM call on the
"give up gracefully" path.
"""

import logging

from common.models import AgentTask, StructuredAgentResult
from services.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EscalationAgent(BaseAgent):
    """
    Fallback agent — triggered when:
      - No other agent is registered for the detected intent.
      - A specialised agent raises EscalationSignal (LLM decided to escalate).
      - A specialised agent throws an unhandled exception.
    """

    SYSTEM_PROMPT = "You are EscalationAgent, a fallback for unresolved support issues."

    def __init__(self, *args, **kwargs):
        super().__init__("EscalationAgent", *args, **kwargs)

    # ------------------------------------------------------------------
    def _build_tools(self) -> list:
        # No tools needed — this agent doesn't run an LLM loop.
        return []

    # ------------------------------------------------------------------
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("[%s] Processing escalation for task %s", self.name, task.task_id)

        escalation_reason = task.params.get(
            "reason", "Issue could not be resolved by automated agents."
        )
        conversation_summary = task.conversation_context[-5:]

        # ── Production integration points (stubbed) ────────────────────
        # ticket_id = self.ecommerce_api_client.create_helpdesk_ticket(...)
        # self.ecommerce_api_client.notify_human_agent(ticket_id=ticket_id)
        ticket_id = f"TKT-{task.task_id[:8].upper()}"
        logger.info("[%s] Mock helpdesk ticket created: %s", self.name, ticket_id)

        return StructuredAgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status="escalation",
            result_data={
                "escalation_reason": escalation_reason,
                "ticket_id": ticket_id,
                "original_query": task.original_query,
                "conversation_summary": conversation_summary,
            },
        )
