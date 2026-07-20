# services/agents/order_tracking_agent.py
"""
Handles "where's my order" style queries: resolves an order_id from the
customer's text, asks the LLM whether/how to call the order-details tool
(services/llm_inference.py call_agent_reason), fetches the row via
EcommerceClient, diagnoses it (call_agent_interpret), then returns a
StructuredOrderSummary for services/orchestrator.py to turn into a reply.

ORDER_ID_PATTERN matches the full `ord-<n>` business-key format this schema
actually uses (db/README.md), preferred over a bare numeric fallback that
never matches any real seeded row on its own. _normalize_order_id() then
recovers that full form even when it didn't come from the regex at all -
see that method's docstring for why that extra step is needed.
"""
import logging
import re
from typing import Callable, Optional # Added Optional

from common.models import AgentTask, StructuredAgentResult, StructuredOrderSummary, LLMAgentReasonRequest, LLMAgentInterpretRequest
from services.agents.base_agent import BaseAgent

# --- Langfuse Integration Start ---
from langfuse import observe
# --- Langfuse Integration End ---

logger = logging.getLogger(__name__)

ORDER_ID_PATTERN = re.compile(r"\bord-\d+\b", re.IGNORECASE)
# Legacy fallback - kept for parity with any caller still passing a bare
# numeric reference ("order 12345"), though this schema's orders.order_id
# is always ord-prefixed, so this alone never matches a real seeded row.
_NUMERIC_ID_PATTERN = re.compile(r"\b(\d{4,})\b")

GET_ORDER_DETAILS_TOOL = "ECommerceAPI.getOrderDetails"
CANCEL_ORDER_TOOL = "ECommerceAPI.cancelOrder"
DELETE_ORDER_TOOL = "ECommerceAPI.deleteOrder"

# Tools whose EcommerceClient guard (order not cancellable/deletable, not
# found) returns a specific reason worth surfacing verbatim - unlike
# getOrderDetails, which keeps the generic "could not find" message below.
_ACTION_TOOLS_WITH_OWN_ERROR_MESSAGE = {CANCEL_ORDER_TOOL, DELETE_ORDER_TOOL}

class OrderTrackingAgent(BaseAgent):
    """
    Specialized AI Agent for handling order tracking inquiries.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("OrderTrackingAgent", *args, **kwargs)

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        """Prefer the full ord-<n> business key; only fall back to a bare
        numeric ID if no business-key-shaped ID is present at all (see
        _NUMERIC_ID_PATTERN's own comment for why that fallback alone never
        actually resolves a real order)."""
        match = ORDER_ID_PATTERN.search(text or "")
        if match:
            return match.group(0).lower()
        match = _NUMERIC_ID_PATTERN.search(text or "")
        return match.group(1) if match else None

    @staticmethod
    def _normalize_order_id(candidate: Optional[str], original_query: str) -> Optional[str]:
        """Recover the full ord-<n> form when `candidate` is just a bare
        number - this happens even when the customer's text clearly says
        "ord-1001", because services/llm_inference.py's router prompt gives
        the LLM a bare-numeric example ({'order_id': '12345'}), which biases
        it to strip the "ord-" prefix when it extracts the ID itself
        (task.params['order_id'], resolved before _extract_order_id's regex
        fallback even runs). This is why the bug was intermittent: it
        depends on the LLM's own extraction, not just the regex. Only ever
        upgrades to a fuller form found verbatim in the customer's own
        message - never invents or guesses an ID."""
        if not candidate:
            return candidate
        if ORDER_ID_PATTERN.match(candidate):
            return candidate.lower()
        if candidate.isdigit():
            match = re.search(rf"\bord-{re.escape(candidate)}\b", original_query or "", re.IGNORECASE)
            if match:
                return match.group(0).lower()
        return candidate

    # (Langfuse decorator would go here, when we add it back)
    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="order_tracking_agent_process_task")
    # --- Langfuse Integration End ---
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("Received task: task_id=%s intent=%s", task.task_id, task.intent)

        user_id = task.user_id
        # Resolve order_id at the very beginning to avoid NameError
        resolved_order_id: Optional[str] = task.params.get('order_id') or self._extract_order_id(task.original_query)
        resolved_order_id = self._normalize_order_id(resolved_order_id, task.original_query)
        logger.info("Resolved order_id=%s user_id=%s", resolved_order_id, user_id)

        # Tool registry:
        tools: dict[str, Callable[[], dict]] = {
            GET_ORDER_DETAILS_TOOL: lambda: self.ecommerce_api_client.get_order_details(user_id, resolved_order_id),
            CANCEL_ORDER_TOOL: lambda: self.ecommerce_api_client.cancel_order(user_id, resolved_order_id),
            DELETE_ORDER_TOOL: lambda: self.ecommerce_api_client.delete_order(user_id, resolved_order_id),
        }
        logger.debug("Available tools for LLM: %s", list(tools.keys()))

        # 1. Use LLMInf_AgentReason for structured workflow planning/tool selection
        reason_response = self.llm_inference_client.call_agent_reason(
            LLMAgentReasonRequest(
                session_id=task.session_id,
                agent_name=self.name,
                task_description=(
                    f"Handle this order-related request for order {resolved_order_id}, "
                    f"customer {user_id}: \"{task.original_query}\""
                ),
                current_state={'order_id': resolved_order_id, 'user_id': user_id},
                available_tools=list(tools.keys()),
            )
        )

        raw_order_details = {}
        invoked_tool_identifier = None
        logger.info(
            "Reasoning result: action=%s tool_name=%s method=%s thought=%s",
            reason_response.action, reason_response.tool_name, 
            reason_response.tool_params.get('method') if reason_response.tool_params else None,
            reason_response.thought,
        )

        if reason_response.action == 'call_api' and reason_response.tool_name and reason_response.tool_params and 'method' in reason_response.tool_params:
            invoked_tool_identifier = f"{reason_response.tool_name}.{reason_response.tool_params['method']}"
            
            if invoked_tool_identifier in tools:
                raw_order_details = tools[invoked_tool_identifier]()
                logger.info("EcommerceClient returned: %s", raw_order_details)
            else:
                logger.warning(
                    "Order lookup skipped - LLM suggested tool '%s' not in agent's registry. Reason: %s",
                    invoked_tool_identifier, reason_response.thought,
                )
        else:
            logger.warning(
                "Order lookup skipped (action=%s, tool_name=%s, method in params=%s) - reason: %s",
                reason_response.action, reason_response.tool_name,
                'method' in reason_response.tool_params if reason_response.tool_params else False,
                reason_response.thought,
            )

        # Graceful fallback: the reasoning step sometimes declines to call any
        # tool at all, or names a plausible-sounding one that was never in the
        # registry (e.g. "getOrderStatus") - both warning branches above.
        # raw_order_details is still {} (its initial value) only in those two
        # cases - a tool that WAS actually invoked always returns a non-empty
        # dict, success or a guarded {"error": ...} (ownership/status checks),
        # so this can never override or bypass those. Defaults to the
        # read-only lookup rather than surfacing a blanket "could not find" to
        # a customer who referenced a real, resolved order_id - always safe
        # (never cancels/deletes on the reasoning step's behalf).
        if not raw_order_details and resolved_order_id and invoked_tool_identifier not in tools:
            logger.info("Falling back to %s for resolved order_id=%s", GET_ORDER_DETAILS_TOOL, resolved_order_id)
            invoked_tool_identifier = GET_ORDER_DETAILS_TOOL
            raw_order_details = tools[invoked_tool_identifier]()

        if not raw_order_details or "error" in raw_order_details:
            # cancelOrder/deleteOrder's guards (already delivered/cancelled, not
            # found, wrong status to delete) return a specific reason worth
            # surfacing; getOrderDetails keeps the existing generic message
            # rather than exposing raw DB wording.
            if invoked_tool_identifier in _ACTION_TOOLS_WITH_OWN_ERROR_MESSAGE and raw_order_details.get("error"):
                message = raw_order_details["error"]
            else:
                message = f"Could not find details for order {resolved_order_id}. Please check the ID or try again."
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="failure",
                result_data={"message": message},
            )

        # 3. Diagnose the order (e.g. shipping delay, payment pending) from the
        # raw DB row via the LLM's interpret step.
        if invoked_tool_identifier == CANCEL_ORDER_TOOL:
            interpretation_goal = "confirm the cancellation and note anything the customer should know"
        elif invoked_tool_identifier == DELETE_ORDER_TOOL:
            interpretation_goal = "confirm the deletion and note anything the customer should know"
        else:
            interpretation_goal = "diagnose order issue"

        interpret_response = self.llm_inference_client.call_agent_interpret(
            LLMAgentInterpretRequest(
                session_id=task.session_id,
                agent_name=self.name,
                raw_data=raw_order_details,
                interpretation_goal=interpretation_goal,
            )
        )

        # 4. Compile all findings into a final structured result.
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
