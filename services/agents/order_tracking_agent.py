# services/agents/order_tracking_agent.py
"""
Handles "where's my order" style queries: resolves an order_id from the
customer's text, asks the LLM whether/how to call the order-details tool
(services/llm_inference.py call_agent_reason), fetches the row via
EcommerceClient, diagnoses it (call_agent_interpret), then returns a
StructuredOrderSummary for services/orchestrator.py to turn into a reply.

createOrder/updateOrder can't resolve "2 red t-shirts" into an item_id on
their own, so searchItems is offered as an optional lookup step first: the
reasoning loop below lets the LLM call searchItems, see the matching catalog
items (with their item_id/price) in the next iteration's current_state, then
call createOrder/updateOrder with the item_id(s) it picked. searchItems is
never treated as the terminal action - the loop keeps going until a terminal
tool (getOrderDetails/cancelOrder/createOrder/updateOrder) runs or the LLM
stops asking for tools.

updateOrder is guarded the same way cancelOrder is (EcommerceClient's
_EDITABLE_STATUSES): a 'delivered', 'cancelled', or 'returned' order can no
longer have its address/items changed.

ORDER_ID_PATTERN only matches purely numeric IDs (e.g. "order 12345"), not the
business-key format this schema actually uses (ord-1001, db/README.md) -
see db/README.md's Known Gaps.
"""
import logging
import re
from typing import Callable, Optional # Added Optional

from common.models import AgentTask, StructuredAgentResult, StructuredOrderSummary, LLMAgentReasonRequest, LLMAgentInterpretRequest
from services.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

ORDER_ID_PATTERN = re.compile(r"\b(\d{4,})\b")

GET_ORDER_DETAILS_TOOL = "ECommerceAPI.getOrderDetails"
CANCEL_ORDER_TOOL = "ECommerceAPI.cancelOrder"
CREATE_ORDER_TOOL = "ECommerceAPI.createOrder"
UPDATE_ORDER_TOOL = "ECommerceAPI.updateOrder"
SEARCH_ITEMS_TOOL = "ECommerceAPI.searchItems"

# Tools whose EcommerceClient guard (order not cancellable/editable,
# customer/item not found, insufficient stock) returns a specific reason
# worth surfacing verbatim - unlike getOrderDetails, which keeps the generic
# "could not find" message below.
_ACTION_TOOLS_WITH_OWN_ERROR_MESSAGE = {CANCEL_ORDER_TOOL, CREATE_ORDER_TOOL, UPDATE_ORDER_TOOL}

# searchItems is a lookup step, not a terminal action - cap how many times the
# LLM can loop (e.g. search, refine search, then createOrder) before giving up
# rather than looping forever on a confused reasoning response.
_MAX_REASONING_STEPS = 3

class OrderTrackingAgent(BaseAgent):
    """
    Specialized AI Agent for handling order tracking inquiries.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("OrderTrackingAgent", *args, **kwargs)

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        match = ORDER_ID_PATTERN.search(text or "")
        return match.group(1) if match else None

    # (Langfuse decorator would go here, when we add it back)
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("Received task: task_id=%s intent=%s", task.task_id, task.intent)

        user_id = task.user_id
        # Resolve order_id at the very beginning to avoid NameError
        resolved_order_id: Optional[str] = task.params.get('order_id') or self._extract_order_id(task.original_query)
        # createOrder has no order_id yet - it needs the line items (and
        # optional shipping address) instead. If the caller didn't already
        # supply them (e.g. a direct API call), the reasoning loop below lets
        # the LLM fill them in itself after a searchItems lookup.
        resolved_line_items: list = task.params.get('line_items') or []
        resolved_shipping_address: Optional[str] = task.params.get('shipping_address')
        # updateOrder edits an existing order's address and/or item
        # quantities - resolved the same way as the fields above, with the
        # LLM's tool_params as a fallback once it's worked out which items to
        # change (optionally after a searchItems lookup).
        resolved_item_updates: list = task.params.get('item_updates') or []
        logger.info("Resolved order_id=%s user_id=%s", resolved_order_id, user_id)

        # Tool registry. Every tool takes the LLM's tool_params dict for that
        # call, even though most ignore it - only searchItems, createOrder,
        # and updateOrder (as a fallback when task.params didn't already
        # supply the relevant fields) actually read anything out of it.
        tools: dict[str, Callable[[dict], dict]] = {
            GET_ORDER_DETAILS_TOOL: lambda params: self.ecommerce_api_client.get_order_details(user_id, resolved_order_id),
            CANCEL_ORDER_TOOL: lambda params: self.ecommerce_api_client.cancel_order(user_id, resolved_order_id),
            CREATE_ORDER_TOOL: lambda params: self.ecommerce_api_client.create_order(
                user_id,
                resolved_line_items or (params or {}).get('line_items') or [],
                resolved_shipping_address or (params or {}).get('shipping_address'),
            ),
            UPDATE_ORDER_TOOL: lambda params: self.ecommerce_api_client.update_order(
                user_id,
                resolved_order_id,
                resolved_shipping_address or (params or {}).get('shipping_address'),
                resolved_item_updates or (params or {}).get('item_updates') or [],
            ),
            SEARCH_ITEMS_TOOL: lambda params: {
                "results": self.ecommerce_api_client.search_items(
                    category=(params or {}).get('category'),
                    keyword=(params or {}).get('keyword'),
                    limit=(params or {}).get('limit', 10),
                )
            },
        }
        logger.debug("Available tools for LLM: %s", list(tools.keys()))

        task_description = (
            f"Handle this order-related request for order {resolved_order_id}, "
            f"customer {user_id}: \"{task.original_query}\". "
            "Call the one ECommerceAPI tool that directly satisfies it: getOrderDetails for a status "
            "check, cancelOrder to cancel (call it directly - it already reports if the order can't be "
            "cancelled, so don't call getOrderDetails first), createOrder to place a new order with the "
            "exact param name `line_items` - a list of {item_id, quantity} objects, e.g. {'method': "
            "'createOrder', 'line_items': [{'item_id': 'item-1030', 'quantity': 1}]} - or updateOrder to "
            "change shipping_address and/or item_updates on an existing order - item_updates is only for "
            "adding/changing/removing an item (same {item_id, quantity} list shape, where quantity 0 "
            "removes that item); omit item_updates entirely for an address-only change, do not invent one. "
            "Never put item_id/quantity as top-level tool_params keys - they always belong inside that "
            "line_items/item_updates list. Any item named but not yet tied to an item_id - whether it's "
            "new to the order or already on it - must be resolved with searchItems first, using the exact "
            "param name `keyword` (e.g. {'method': 'searchItems', 'keyword': 'red t-shirt'} - never "
            "`query` or `search_term`). Once `search_results` in Current state already contains the item "
            "you need, do NOT call searchItems again - your very next action must be createOrder/"
            "updateOrder using that item's item_id."
        )
        current_state: dict = {'order_id': resolved_order_id, 'user_id': user_id}

        # 1. Use LLMInf_AgentReason for structured workflow planning/tool selection.
        # Looped so an optional searchItems call can inform a following
        # createOrder call - searchItems is never itself the terminal action.
        raw_order_details = {}
        invoked_tool_identifier = None
        for step in range(_MAX_REASONING_STEPS):
            reason_response = self.llm_inference_client.call_agent_reason(
                LLMAgentReasonRequest(
                    session_id=task.session_id,
                    agent_name=self.name,
                    task_description=task_description,
                    current_state=current_state,
                    available_tools=list(tools.keys()),
                )
            )
            logger.info(
                "Reasoning step=%d result: action=%s tool_name=%s method=%s thought=%s",
                step, reason_response.action, reason_response.tool_name,
                reason_response.tool_params.get('method') if reason_response.tool_params else None,
                reason_response.thought,
            )

            if not (reason_response.action == 'call_api' and reason_response.tool_name and reason_response.tool_params and 'method' in reason_response.tool_params):
                logger.warning(
                    "Order lookup stopped (action=%s, tool_name=%s, method in params=%s) - reason: %s",
                    reason_response.action, reason_response.tool_name,
                    'method' in reason_response.tool_params if reason_response.tool_params else False,
                    reason_response.thought,
                )
                break

            tool_identifier = f"{reason_response.tool_name}.{reason_response.tool_params['method']}"

            if tool_identifier not in tools:
                logger.warning(
                    "Order lookup stopped - LLM suggested tool '%s' not in agent's registry. Reason: %s",
                    tool_identifier, reason_response.thought,
                )
                break

            tool_result = tools[tool_identifier](reason_response.tool_params)
            logger.info("EcommerceClient returned: %s", tool_result)

            if tool_identifier == SEARCH_ITEMS_TOOL:
                # A lookup step, not terminal - feed the results back in so the
                # next reasoning step can pick item_id(s) for createOrder.
                current_state['search_results'] = tool_result['results']
                continue

            invoked_tool_identifier = tool_identifier
            raw_order_details = tool_result
            break

        if not raw_order_details or "error" in raw_order_details:
            # cancelOrder/createOrder's guards (already delivered/cancelled,
            # customer/item not found, insufficient stock) return a specific
            # reason worth surfacing; getOrderDetails keeps the existing
            # generic message rather than exposing raw DB wording.
            if invoked_tool_identifier in _ACTION_TOOLS_WITH_OWN_ERROR_MESSAGE and raw_order_details.get("error"):
                message = raw_order_details["error"]
            elif current_state.get('search_results') == []:
                message = "Couldn't find any items matching your request. Please try different keywords."
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
        elif invoked_tool_identifier == CREATE_ORDER_TOOL:
            interpretation_goal = "confirm the order placement and note anything the customer should know"
        elif invoked_tool_identifier == UPDATE_ORDER_TOOL:
            interpretation_goal = "confirm the order update and note anything the customer should know"
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
