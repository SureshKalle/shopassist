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

Confirmation gate (cancel/delete): CANCEL_ORDER_TOOL/DELETE_ORDER_TOOL never
execute on the LLM reasoning step's decision alone - process_task() asks the
customer to confirm first (result_data.needs_confirmation=True), and
services/orchestrator.py remembers what was pending, keyed by session_id.
The customer's next reply comes back here via _handle_confirmation_reply()
(orchestrator.py routes it directly, bypassing the normal decompose/reason
flow - see that file's own comment) instead of a second LLM decision, using
a fixed, deterministic keyword check rather than another LLM call: same
"auditable, no-model-in-the-loop" philosophy services/guardrails.py already
uses for its safety checks, and specifically avoids a second model call
being able to talk itself into confirming an action it was never actually
asked to confirm. An ambiguous reply fails closed - the action is not
executed, same as an explicit decline.
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

# Destructive tools that must not execute on a single LLM reasoning decision -
# see the module docstring's "Confirmation gate" section above.
_DESTRUCTIVE_TOOLS = {CANCEL_ORDER_TOOL, DELETE_ORDER_TOOL}
_ACTION_VERB = {CANCEL_ORDER_TOOL: "cancel", DELETE_ORDER_TOOL: "delete"}

# Anchored at the start of the reply - customers confirming/declining rarely
# bury it mid-sentence, and anchoring avoids a stray "yes" appearing later
# in a longer, unrelated reply being misread as confirmation. Deliberately
# narrow, not a general sentiment classifier - anything that matches neither
# pattern is ambiguous and fails closed (see _handle_confirmation_reply).
_AFFIRMATIVE_PATTERN = re.compile(
    r"^\s*(yes|yeah|yep|yup|confirm(?:ed)?|go ahead|do it|please do|sure|ok(?:ay)?)\b", re.IGNORECASE
)
_NEGATIVE_PATTERN = re.compile(
    r"^\s*(no|nope|nah|cancel that|never\s*mind|nevermind|don'?t|stop)\b", re.IGNORECASE
)


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

    def _build_tools(self, user_id: str, order_id: Optional[str]) -> dict[str, Callable[[], dict]]:
        """The order-action tool registry - built the same way for both the
        normal reasoning path (process_task) and the post-confirmation
        execution path (_handle_confirmation_reply) below, so the two can
        never drift apart."""
        return {
            GET_ORDER_DETAILS_TOOL: lambda: self.ecommerce_api_client.get_order_details(user_id, order_id),
            CANCEL_ORDER_TOOL: lambda: self.ecommerce_api_client.cancel_order(user_id, order_id),
            DELETE_ORDER_TOOL: lambda: self.ecommerce_api_client.delete_order(user_id, order_id),
        }

    def _finalize_result(
        self, task: AgentTask, invoked_tool_identifier: Optional[str],
        raw_order_details: dict, resolved_order_id: Optional[str],
    ) -> StructuredAgentResult:
        """Shared tail for both process_task()'s normal reasoning path and
        _handle_confirmation_reply()'s post-confirmation execution path: turn
        a tool's raw result (or lack of one) into the StructuredAgentResult
        services/orchestrator.py expects. This is exactly what process_task()
        always did inline before the confirmation gate needed a second entry
        point into the same finishing logic.
        """
        # Graceful fallback: the reasoning step sometimes declines to call any
        # tool at all, or names a plausible-sounding one that was never in the
        # registry (e.g. "getOrderStatus"). raw_order_details is still {} (its
        # initial value) only in those cases - a tool that WAS actually
        # invoked always returns a non-empty dict, success or a guarded
        # {"error": ...} (ownership/status checks), so this can never override
        # or bypass those.
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

        # Diagnose the order (e.g. shipping delay, payment pending) from the
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

    def _handle_confirmation_reply(self, task: AgentTask) -> StructuredAgentResult:
        """The turn after process_task() below asked "are you sure?" -
        task.params carries exactly what was pending (services/orchestrator.py
        stashes result_data['pending_confirmation'] keyed by session_id when
        that happens, and hands it back unchanged as this task's params - see
        that file's own comment on the confirmation gate). See
        _AFFIRMATIVE_PATTERN/_NEGATIVE_PATTERN above for why this is a fixed
        keyword check, not a second LLM call. Ambiguous replies fail closed:
        the action is NOT executed, same as an explicit decline - a
        destructive action only ever proceeds on an unambiguous "yes".
        """
        tool_name = task.params.get("tool")
        order_id = task.params.get("order_id")
        verb = _ACTION_VERB.get(tool_name, "process")
        reply = (task.original_query or "").strip()

        if _AFFIRMATIVE_PATTERN.match(reply):
            logger.info(
                "OrderTrackingAgent: confirmation received - executing %s for order_id=%s", tool_name, order_id
            )
            tools = self._build_tools(task.user_id, order_id)
            if tool_name in tools:
                raw_order_details = tools[tool_name]()
            else:
                # Only reachable if pending_confirmation was ever stashed with
                # something other than the two known destructive tools -
                # doesn't happen via the path that sets it (see process_task()
                # below), but fail safe rather than KeyError if it ever did.
                logger.error("OrderTrackingAgent: pending confirmation named unknown tool %r - not executing", tool_name)
                raw_order_details = {"error": "Could not resolve which action to confirm - please ask again."}
            return self._finalize_result(task, tool_name, raw_order_details, order_id)

        if _NEGATIVE_PATTERN.match(reply):
            logger.info("OrderTrackingAgent: customer declined - not executing %s for order_id=%s", tool_name, order_id)
            message = f"No problem - order {order_id} was left unchanged."
        else:
            logger.info(
                "OrderTrackingAgent: ambiguous confirmation reply (%r) - failing closed, not executing %s for order_id=%s",
                reply, tool_name, order_id,
            )
            message = (
                f"I didn't catch a clear yes or no, so I haven't touched order {order_id} - "
                f"let me know if you'd still like me to {verb} it."
            )

        return StructuredAgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status="success",
            result_data={"message": message},
        )

    # (Langfuse decorator would go here, when we add it back)
    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="order_tracking_agent_process_task")
    # --- Langfuse Integration End ---
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("Received task: task_id=%s intent=%s", task.task_id, task.intent)

        # A reply to a previously-asked "are you sure you want to
        # cancel/delete order X?" - services/orchestrator.py routes it here
        # directly (see that file's own confirmation-gate comment) instead of
        # through the normal decompose/reason flow below, since a bare
        # "yes"/"no" isn't itself a fresh order-tracking request to route.
        if task.params.get("confirm_pending_action"):
            return self._handle_confirmation_reply(task)

        user_id = task.user_id
        # Resolve order_id at the very beginning to avoid NameError
        resolved_order_id: Optional[str] = task.params.get('order_id') or self._extract_order_id(task.original_query)
        resolved_order_id = self._normalize_order_id(resolved_order_id, task.original_query)
        logger.debug("OrderTrackingAgent: Initial resolved_order_id=%s from task.params/query", resolved_order_id)
        logger.info("Resolved order_id=%s user_id=%s", resolved_order_id, user_id)

        # Tool registry:
        tools = self._build_tools(user_id, resolved_order_id)
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
                if invoked_tool_identifier in _DESTRUCTIVE_TOOLS:
                    # Confirmation gate: don't execute a destructive action on
                    # a single LLM reasoning decision - ask first, and stash
                    # what would have run so services/orchestrator.py can hand
                    # it straight back to _handle_confirmation_reply() above
                    # if the customer confirms next turn.
                    verb = _ACTION_VERB[invoked_tool_identifier]
                    logger.info(
                        "OrderTrackingAgent: destructive action '%s' requested for order_id=%s - "
                        "asking for confirmation instead of executing",
                        verb, resolved_order_id,
                    )
                    return StructuredAgentResult(
                        task_id=task.task_id,
                        agent_name=self.name,
                        status="success",
                        result_data={
                            "message": (
                                f"Just to confirm - you'd like me to {verb} order {resolved_order_id}? "
                                "Reply \"yes\" to confirm, or \"no\" to leave it as is."
                            ),
                            "needs_confirmation": True,
                            "pending_confirmation": {"tool": invoked_tool_identifier, "order_id": resolved_order_id},
                        },
                    )
                raw_order_details = tools[invoked_tool_identifier]()
                logger.debug("OrderTrackingAgent: Raw order details from ECommerceClient: %s", raw_order_details)
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

        return self._finalize_result(task, invoked_tool_identifier, raw_order_details, resolved_order_id)
