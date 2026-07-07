# services/agents/returns_agent.py
"""
ReturnsAgent — LangChain tool-calling edition.

Tools:
  - search_return_policy : RAG retriever scoped to customer_support_policy docs
  - get_order_details     : e-commerce API, used to check eligibility windows

The LLM decides whether it needs order details (e.g. to check ship date
against the return window) before answering.
"""

import logging

from langchain_core.tools import tool

from common.models import AgentTask, StructuredAgentResult
from services.agents.base_agent import BaseAgent, EscalationSignal

logger = logging.getLogger(__name__)


class ReturnsAgent(BaseAgent):
    """Specialised agent for returns, refunds, exchanges, and cancellations."""

    SYSTEM_PROMPT = (
        "You are ReturnsAgent, a customer-support specialist for returns, "
        "refunds, exchanges, and cancellations.\n\n"
        "Tools available:\n"
        "  - search_return_policy(query): look up relevant policy text\n"
        "  - get_order_details(customer_id, order_id): check an order's status, "
        "useful for eligibility checks\n\n"
        "Always check the policy first. If the customer mentions a specific order, "
        "also check its details to determine eligibility (e.g. electronics have a "
        "shorter return window). Give a clear, empathetic, actionable answer. "
        "If you cannot determine eligibility or the situation is unusual (e.g. "
        "damaged/lost item, customer is upset), respond with exactly: "
        "'ESCALATE: <brief reason>'."
    )

    def __init__(self, *args, **kwargs):
        super().__init__("ReturnsAgent", *args, **kwargs)

    # ------------------------------------------------------------------
    def _build_tools(self) -> list:
        ecommerce_client = self.ecommerce_api_client
        retriever = self.rag_service.as_retriever(
            search_kwargs={"k": 2, "filter": {"source_type": "customer_support_policy"}}
        )

        @tool
        def search_return_policy(query: str) -> str:
            """Search the return/refund/warranty policy knowledge base."""
            docs = retriever.invoke(query)
            if not docs:
                return "No specific policy found for this query."
            return "\n---\n".join(d.page_content for d in docs)

        @tool
        def get_order_details(customer_id: str, order_id: str) -> dict:
            """Fetch order details to check return eligibility (e.g. ship date, item type)."""
            return ecommerce_client.get_order_details(customer_id, order_id)

        return [search_return_policy, get_order_details]

    # ------------------------------------------------------------------
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("[%s] Processing task %s", self.name, task.task_id)

        order_id = task.params.get("order_id") or self._extract_order_id(task.original_query)

        if self._agent_graph is None:
            return self._mock_process(task, order_id)

        instruction = (
            f"Customer ID: {task.customer_id}\n"
            f"Order ID (if known): {order_id or 'not provided'}\n"
            f"Customer's request: '{task.original_query}'\n\n"
            "Help the customer with their return/refund/exchange request."
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
            return self._mock_process(task, order_id)

        answer = agent_result["final_text"] or (
            "I can help you start a return or refund. Could you share your order ID?"
        )
        result_data = {"answer_snippet": answer}
        if order_id:
            result_data["order_id"] = order_id

        return StructuredAgentResult(
            task_id=task.task_id, agent_name=self.name, status="success",
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    def _mock_process(self, task: AgentTask, order_id: str | None) -> StructuredAgentResult:
        """Deterministic fallback path used when no LLM/API key is available."""
        retriever = self.rag_service.as_retriever(
            search_kwargs={"k": 1, "filter": {"source_type": "customer_support_policy"}}
        )
        docs = retriever.invoke(f"return refund policy {task.original_query}")
        if not docs:
            docs = self.rag_service.as_retriever(search_kwargs={"k": 1}).invoke(task.original_query)

        answer = (
            docs[0].page_content
            if docs
            else "I can help you start a return or refund. Could you share your order ID?"
        )
        result_data = {"answer_snippet": answer}
        if order_id:
            result_data["order_id"] = order_id

        return StructuredAgentResult(
            task_id=task.task_id, agent_name=self.name, status="success",
            result_data=result_data,
        )
