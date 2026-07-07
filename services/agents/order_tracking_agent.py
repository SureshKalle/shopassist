# services/agents/order_tracking_agent.py
"""
OrderTrackingAgent — LangChain tool-calling edition.

The agent is given one tool (`get_order_details`) and a system prompt
instructing it to fetch order data, diagnose any issue, and answer.
The LLM decides on its own whether/when to call the tool — no hand-rolled
ReAct loop needed.
"""

import logging

from langchain_core.tools import tool

from common.models import AgentTask, StructuredAgentResult, StructuredOrderSummary
from services.agents.base_agent import BaseAgent, EscalationSignal

logger = logging.getLogger(__name__)


class OrderTrackingAgent(BaseAgent):
    """Specialised agent for order tracking and shipping inquiries."""

    SYSTEM_PROMPT = (
        "You are OrderTrackingAgent, a customer-support specialist for order "
        "status, shipping, and delivery questions.\n\n"
        "You have a `get_order_details` tool. Use it to fetch the order's status, "
        "items, and estimated delivery. Then:\n"
        "  - If status is 'Pending', flag that payment may need attention.\n"
        "  - If status is 'Shipped', confirm delivery ETA.\n"
        "  - If the order is not found, say so clearly and ask the customer to "
        "double check the order ID.\n\n"
        "Respond with a concise, factual summary of the order status suitable "
        "for a downstream response-generation step (not the final customer-facing "
        "message itself — just the facts).\n\n"
        "If you cannot resolve the issue at all (e.g. repeated tool failures), "
        "respond with exactly: 'ESCALATE: <brief reason>'."
    )

    def __init__(self, *args, **kwargs):
        super().__init__("OrderTrackingAgent", *args, **kwargs)

    # ------------------------------------------------------------------
    def _build_tools(self) -> list:
        ecommerce_client = self.ecommerce_api_client

        @tool
        def get_order_details(customer_id: str, order_id: str) -> dict:
            """Fetch order details (status, items, estimated delivery) for a given
            customer_id and order_id from the e-commerce order management system."""
            return ecommerce_client.get_order_details(customer_id, order_id)

        return [get_order_details]

    # ------------------------------------------------------------------
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("[%s] Processing task %s", self.name, task.task_id)

        customer_id = task.customer_id
        order_id = (
            task.params.get("order_id")
            or self._extract_order_id(task.original_query)
            or "12345"
        )

        if self._agent_graph is None:
            return self._mock_process(task, customer_id, order_id)

        instruction = (
            f"Customer ID: {customer_id}\n"
            f"Order ID: {order_id}\n"
            f"Customer's message: '{task.original_query}'\n\n"
            "Look up this order and summarise its status, items, and delivery estimate."
        )

        try:
            agent_result = self.run_agent(task, instruction)
        except EscalationSignal as esc:
            return StructuredAgentResult(
                task_id=task.task_id, agent_name=self.name, status="escalation",
                result_data={"escalation_reason": esc.reason},
            )
        except RuntimeError as exc:
            logger.warning("[%s] Agent graph unavailable (%s) — using mock.", self.name, exc)
            return self._mock_process(task, customer_id, order_id)

        # Pull the raw tool output (get_order_details) from the trace
        raw_order = {}
        for call in agent_result["tool_calls"]:
            if call["tool"] == "get_order_details":
                raw_order = call["output"] if isinstance(call["output"], dict) else {}

        if not raw_order:
            # Fall back to a direct call if the LLM never invoked the tool
            raw_order = self.ecommerce_api_client.get_order_details(customer_id, order_id)

        if not raw_order or "error" in raw_order:
            return StructuredAgentResult(
                task_id=task.task_id, agent_name=self.name, status="failure",
                result_data={"message": f"Could not find details for order {order_id}."},
            )

        order_summary = StructuredOrderSummary(
            order_id=raw_order.get("order_id", order_id),
            status=raw_order.get("status", "Unknown"),
            items=raw_order.get("items", []),
            estimated_delivery=raw_order.get("estimated_delivery"),
            issue_analysis=agent_result["final_text"] or None,
        )

        return StructuredAgentResult(
            task_id=task.task_id, agent_name=self.name, status="success",
            result_data=order_summary,
        )

    # ------------------------------------------------------------------
    def _mock_process(
        self, task: AgentTask, customer_id: str, order_id: str
    ) -> StructuredAgentResult:
        """Deterministic fallback path used when no LLM/API key is available."""
        raw_order = self.ecommerce_api_client.get_order_details(customer_id, order_id)
        if not raw_order or "error" in raw_order:
            return StructuredAgentResult(
                task_id=task.task_id, agent_name=self.name, status="failure",
                result_data={"message": f"Could not find details for order {order_id}."},
            )

        issue_analysis = None
        if raw_order.get("status") == "Pending":
            issue_analysis = "PaymentPending — Check payment method"
        elif raw_order.get("status") == "Shipped":
            issue_analysis = "Order is on its way"

        order_summary = StructuredOrderSummary(
            order_id=raw_order.get("order_id", order_id),
            status=raw_order.get("status", "Unknown"),
            items=raw_order.get("items", []),
            estimated_delivery=raw_order.get("estimated_delivery"),
            issue_analysis=issue_analysis,
        )
        return StructuredAgentResult(
            task_id=task.task_id, agent_name=self.name, status="success",
            result_data=order_summary,
        )
