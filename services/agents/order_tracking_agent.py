# services/agents/order_tracking_agent.py
"""
Handles "where's my order" style queries: resolves an order_id from the
customer's text, asks the LLM whether/how to call the order-details tool
(services/llm_inference.py call_agent_reason), fetches the row via
EcommerceClient, diagnoses it (call_agent_interpret), then returns a
StructuredOrderSummary for services/orchestrator.py to turn into a reply.

ORDER_ID_PATTERN only matches purely numeric IDs (e.g. "order 12345"), not the
business-key format this schema actually uses (ord-1001, db/README.md) -
see db/README.md's Known Gaps.
"""
import logging
import re
from typing import Callable

from common.models import AgentTask, StructuredAgentResult, StructuredOrderSummary, LLMAgentReasonRequest, LLMAgentInterpretRequest
from services.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

ORDER_ID_PATTERN = re.compile(r"\b(\d{4,})\b")

GET_ORDER_DETAILS_TOOL = "ECommerceAPI.getOrderDetails"
CANCEL_ORDER_TOOL = "ECommerceAPI.cancelOrder"

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

    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("Received task: task_id=%s intent=%s", task.task_id, task.intent)

        user_id = task.user_id  # the logged-in identifier, sent by shopassist-client (see common/models.py CustomerQuery)
        # The router never actually populates 'order_id' into task.params (it only
        # passes {"query": <text>}), so pull it out of the customer's own text instead
        # of silently defaulting to '12345' for every order.
        order_id = task.params.get('order_id') or self._extract_order_id(task.original_query)
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
        if reason_response.action == 'call_api' and reason_response.tool_name in tools:
            # 2. Execute Internal Tool: E-commerce Microservice API call
            invoked_tool = reason_response.tool_name
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
