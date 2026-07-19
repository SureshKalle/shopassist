# services/agents/order_tracking_agent.py
"""
Handles "where's my order" style queries: resolves an order_id from the
customer's text, asks the LLM whether/how to call the order-details tool
(services/llm_inference.py call_agent_reason), fetches the row via
EcommerceClient, diagnoses it (call_agent_interpret), then returns a
StructuredOrderSummary for services/orchestrator.py to turn into a reply.

ORDER_ID_PATTERN matches the business-key format this schema actually uses
(ord-1001, db/README.md) first, falling back to a bare numeric ID (e.g.
"order 12345") if no ord-prefixed one is present - previously it matched
*only* the bare-numeric form, so "order ord-1001" resolved to "1001", which
matches no row (orders.order_id is always the full "ord-1001" string) and
every real lookup failed before ever reaching EcommerceClient.
"""
import logging
import re
from typing import Callable

from common.models import AgentTask, Message, StructuredAgentResult, StructuredOrderSummary, LLMAgentReasonRequest, LLMAgentInterpretRequest
from services.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

ORDER_ID_PATTERN = re.compile(r"\b(ord-\d+|\d{4,})\b", re.IGNORECASE)

GET_ORDER_DETAILS_TOOL = "ECommerceAPI.getOrderDetails"
CANCEL_ORDER_TOOL = "ECommerceAPI.cancelOrder"

# Guardrail: cancellation is a destructive action, so it must never happen on
# a single unreviewed LLM judgement call alone (the same call_agent_reason
# step that can also get the tool name itself wrong - see _resolve_tool_name
# above). The marker is embedded in our own clarifying question below and
# checked for on the *next* turn via conversation_context, so a bare "cancel
# my order" first message can never slip through as if it were already a
# confirmation - only an explicit reply to our own prompt counts.
CANCEL_CONFIRMATION_MARKER = "reply 'yes' to confirm the cancellation"
_AFFIRMATIVE_WORDS = ("yes", "yeah", "yep", "confirm", "confirmed", "sure", "go ahead", "correct", "please do")


def _awaiting_cancel_confirmation(conversation_context: list[Message]) -> bool:
    for msg in reversed(conversation_context or []):
        if msg.role == "assistant":
            return CANCEL_CONFIRMATION_MARKER in msg.content.lower()
    return False


def _is_affirmative(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(lowered == word or lowered.startswith(word) for word in _AFFIRMATIVE_WORDS)


def _extract_order_id_from_history(conversation_context: list[Message]) -> str | None:
    """Recovers the order_id when a confirmation reply ("yes") doesn't repeat
    it - looks for the id we ourselves quoted in the confirmation prompt."""
    for msg in reversed(conversation_context or []):
        match = ORDER_ID_PATTERN.search(msg.content or "")
        if match:
            return match.group(1).lower()
    return None


def _resolve_tool_name(returned_name: str | None, thought: str, tools: dict) -> str | None:
    """Match the LLM's reasoning output against the registered tool names.

    call_agent_reason's own docstring already flags this as fragile:
    "Callers must compare response.tool_name against the exact string they
    advertised in available_tools ... not just 'ECommerceAPI'" - and in
    practice, local models (confirmed with llama3.2) do exactly that:
    return the abbreviated 'ECommerceAPI' instead of the full
    'ECommerceAPI.getOrderDetails'. The exact-match check that used to gate
    every tool call here (`reason_response.tool_name in tools`) then never
    matches, so the tool never runs and EcommerceClient never even sees a
    query - not a database problem, but indistinguishable from one to a
    customer ("could not find order").

    Falls back to a prefix match; when that's still ambiguous (both this
    agent's tools share the "ECommerceAPI" prefix), breaks the tie using
    the reasoning thought for a cancellation-specific keyword, defaulting
    to the more common get-details tool otherwise.
    """
    name = (returned_name or "").strip()
    if name in tools:
        return name
    candidates = [t for t in tools if name and t.startswith(name)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        if "cancel" in (thought or "").lower() and CANCEL_ORDER_TOOL in candidates:
            return CANCEL_ORDER_TOOL
        if GET_ORDER_DETAILS_TOOL in candidates:
            return GET_ORDER_DETAILS_TOOL
        return candidates[0]
    return None


class OrderTrackingAgent(BaseAgent):
    """
    Specialized AI Agent for handling order tracking inquiries.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("OrderTrackingAgent", *args, **kwargs)

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        match = ORDER_ID_PATTERN.search(text or "")
        return match.group(1).lower() if match else None

    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("Received task: task_id=%s intent=%s", task.task_id, task.intent)

        user_id = task.user_id  # the logged-in identifier, sent by shopassist-client (see common/models.py CustomerQuery)
        # The router never actually populates 'order_id' into task.params (it only
        # passes {"query": <text>}), so pull it out of the customer's own text instead
        # of silently defaulting to '12345' for every order.
        order_id = task.params.get('order_id') or self._extract_order_id(task.original_query)

        awaiting_confirmation = _awaiting_cancel_confirmation(task.conversation_context)
        if order_id is None and awaiting_confirmation:
            # A bare "yes" reply to our own confirmation prompt won't contain
            # the order_id - recover it from the prompt we asked it in.
            order_id = _extract_order_id_from_history(task.conversation_context)
        logger.info("Resolved order_id=%s user_id=%s", order_id, user_id)

        # Tool registry: the single source of truth for both what's advertised
        # to the LLM (available_tools below) and what actually runs (the
        # dispatch after call_agent_reason) - a tool can't be offered without
        # being invocable, or invoked without being offered. Add a new tool by
        # adding one entry here, not by hand-syncing a string in two places.
        tools: dict[str, Callable[[], dict]] = {
            GET_ORDER_DETAILS_TOOL: lambda: self.ecommerce_api_client.get_order_details(user_id, order_id),
            CANCEL_ORDER_TOOL: lambda: self.ecommerce_api_client.cancel_order(user_id, order_id),
        }

        # 1. Use LLMInf_AgentReason for structured workflow planning/tool selection
        # (This determines if we need to call an API, RAG, or return directly)
        reason_response = self.llm_inference_client.call_agent_reason(
            LLMAgentReasonRequest(
                session_id=task.session_id,
                agent_name=self.name,
                task_description=(
                    f"Handle this order-related request for order {order_id}, "
                    f"customer {user_id}: \"{task.original_query}\""
                ),
                current_state={'order_id': order_id, 'user_id': user_id},
                available_tools=list(tools), # Inform the LLM of available tools
            )
        )

        raw_order_details = {}
        invoked_tool = None
        logger.info(
            "Reasoning result: action=%s tool_name=%s thought=%s",
            reason_response.action, reason_response.tool_name, reason_response.thought,
        )
        resolved_tool = _resolve_tool_name(reason_response.tool_name, reason_response.thought, tools)
        confirmed_now = awaiting_confirmation and _is_affirmative(task.original_query)

        if confirmed_now:
            # The customer is replying "yes" to our own confirmation prompt,
            # not describing their request from scratch - call_agent_reason
            # has no memory of *why* it was asked (its task_description above
            # doesn't include conversation history), so a bare "yes" gives it
            # nothing to reason from and it tends to pick a harmless default
            # like getOrderDetails instead. The confirmation state itself
            # already answers "what should happen next": proceed with the
            # cancellation the customer just confirmed, rather than asking
            # the reasoning step to re-derive that from "yes" alone.
            resolved_tool = CANCEL_ORDER_TOOL

        if resolved_tool == CANCEL_ORDER_TOOL and not confirmed_now:
            # Guardrail: never cancel on the LLM's reasoning step alone - it's
            # a destructive action gated on a single unreviewed judgement call
            # (from a model that can also mis-name the tool itself, see
            # _resolve_tool_name). Ask the customer to explicitly confirm on
            # the next turn instead of cancelling immediately.
            logger.info("Cancellation requires confirmation - asking before calling cancel_order for order_id=%s", order_id)
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="success",
                result_data={
                    "message": f"Just to confirm - you'd like to cancel order {order_id}? "
                               f"{CANCEL_CONFIRMATION_MARKER.capitalize()}."
                },
            )

        if (reason_response.action == 'call_api' or confirmed_now) and resolved_tool:
            # 2. Execute Internal Tool: E-commerce Microservice API call
            invoked_tool = resolved_tool
            raw_order_details = tools[invoked_tool]()
            logger.info("EcommerceClient returned: %s", raw_order_details)
        else:
            logger.warning(
                "Order lookup skipped (action=%s, tool_name=%s) - reason: %s",
                reason_response.action, reason_response.tool_name, reason_response.thought,
            )

        if not raw_order_details or "error" in raw_order_details:
            # cancelOrder's guard (already delivered/cancelled, not found) returns
            # a specific reason worth surfacing; getOrderDetails keeps the
            # existing generic message rather than exposing raw DB wording.
            if invoked_tool == CANCEL_ORDER_TOOL and raw_order_details.get("error"):
                message = raw_order_details["error"]
            else:
                message = f"Could not find details for order {order_id}. Please check the ID or try again."
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="failure",
                result_data={"message": message},
            )

        # 3. Diagnose the order (e.g. shipping delay, payment pending) from the
        # raw DB row via the LLM's interpret step.
        interpret_response = self.llm_inference_client.call_agent_interpret(
            LLMAgentInterpretRequest(
                session_id=task.session_id,
                agent_name=self.name,
                raw_data=raw_order_details,
                interpretation_goal=(
                    "confirm the cancellation and note anything the customer should know"
                    if invoked_tool == CANCEL_ORDER_TOOL
                    else "diagnose order issue"
                ),
            )
        )

        # 4. Compile all findings into a final structured result.
        # structured_interpretation is always an OrderIssueAnalysis model (see
        # common/models.py LLMAgentInterpretResponse / services/llm_inference.py
        # call_agent_interpret), never a plain dict - access issue_type directly.
        order_summary = StructuredOrderSummary(
            order_id=raw_order_details.get('order_id'),
            status=raw_order_details.get('status'),
            items=raw_order_details.get('items', []),
            estimated_delivery=raw_order_details.get('estimated_delivery'),
            issue_analysis=interpret_response.structured_interpretation.issue_type,
        )
        
        return StructuredAgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status="success",
            result_data=order_summary,
        )

# This agent would not typically be run directly but instantiated by the Orchestrator
if __name__ == "__main__":
    print("OrderTrackingAgent is a specialized agent.")
    # To test, you would need to mock all its dependencies.
