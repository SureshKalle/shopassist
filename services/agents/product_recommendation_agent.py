# services/agents/product_recommendation_agent.py
"""
ProductRecommendationAgent — LangChain tool-calling edition.

Tools given to the agent:
  - get_customer_history : e-commerce API lookup
  - search_products       : wraps RAGService.as_retriever() as an agent tool

The LLM decides how many times to search, refines its own queries, and
produces the final recommendation text — no hand-rolled orchestration.
"""

import logging
import re

from langchain_core.tools import tool

from common.models import (
    AgentTask,
    StructuredAgentResult,
    StructuredProductRecommendation,
)
from services.agents.base_agent import BaseAgent, EscalationSignal

logger = logging.getLogger(__name__)


class ProductRecommendationAgent(BaseAgent):
    """Specialised agent for personalised product recommendations."""

    SYSTEM_PROMPT = (
        "You are ProductRecommendationAgent, a customer-support specialist for "
        "product suggestions.\n\n"
        "You have two tools:\n"
        "  - get_customer_history(customer_id): past purchases & favourite category\n"
        "  - search_products(query): search the product knowledge base\n\n"
        "Workflow: look up the customer's history, then search products using "
        "terms from both their history and their current request. Pick the single "
        "best-matching product and explain briefly why it fits.\n\n"
        "Respond with the product name, price (if known), and a one-sentence reason. "
        "If no suitable product is found after searching, respond with exactly: "
        "'ESCALATE: <brief reason>'."
    )

    def __init__(self, *args, **kwargs):
        super().__init__("ProductRecommendationAgent", *args, **kwargs)

    # ------------------------------------------------------------------
    def _build_tools(self) -> list:
        ecommerce_client = self.ecommerce_api_client
        retriever = self.rag_service.as_retriever(
            search_kwargs={"k": 5, "filter": {"source_type": "product_catalog"}}
        )

        @tool
        def get_customer_history(customer_id: str) -> dict:
            """Fetch a customer's last purchase and favourite category."""
            return ecommerce_client.get_customer_history(customer_id)

        @tool
        def search_products(query: str) -> str:
            """Search the product knowledge base for items matching the query.
            Returns concatenated product descriptions with pricing/spec info."""
            docs = retriever.invoke(query)
            if not docs:
                return "No matching products found."
            return "\n---\n".join(d.page_content for d in docs)

        return [get_customer_history, search_products]

    # ------------------------------------------------------------------
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("[%s] Processing task %s", self.name, task.task_id)

        customer_id = task.customer_id

        if self._agent_graph is None:
            return self._mock_process(task, customer_id)

        instruction = (
            f"Customer ID: {customer_id}\n"
            f"Customer's request: '{task.original_query}'\n\n"
            "Find the best product recommendation for this customer."
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
            return self._mock_process(task, customer_id)

        final_text = agent_result["final_text"]
        if not final_text or "no matching products" in final_text.lower():
            return StructuredAgentResult(
                task_id=task.task_id, agent_name=self.name, status="failure",
                result_data={"message": "Could not find a suitable product recommendation."},
            )

        # Best-effort structured extraction from the agent's free-text answer
        price_match = re.search(r"\$\s?(\d+(?:\.\d{2})?)", final_text)
        price = float(price_match.group(1)) if price_match else 0.0

        recommendation = StructuredProductRecommendation(
            product_id="see_description",
            name=self._extract_product_name(final_text),
            description_snippet=final_text[:150],
            price=price,
            reason=final_text[:200],
        )
        return StructuredAgentResult(
            task_id=task.task_id, agent_name=self.name, status="success",
            result_data=recommendation,
        )

    # ------------------------------------------------------------------
    def _mock_process(self, task: AgentTask, customer_id: str) -> StructuredAgentResult:
        """Deterministic fallback path used when no LLM/API key is available."""
        history = self.ecommerce_api_client.get_customer_history(customer_id)
        if "error" in history:
            history = {}
        fav_category = history.get("favorite_category", "")
        last_purchase = history.get("last_purchase", "")

        query = self._extract_product_query(task.original_query)
        rag_query = f"{query} {fav_category} {last_purchase}".strip()

        retriever = self.rag_service.as_retriever(
            search_kwargs={"k": 3, "filter": {"source_type": "product_catalog"}}
        )
        docs = retriever.invoke(rag_query)
        if not docs:
            docs = self.rag_service.as_retriever(search_kwargs={"k": 3}).invoke(rag_query)

        if not docs:
            return StructuredAgentResult(
                task_id=task.task_id, agent_name=self.name, status="failure",
                result_data={"message": "Could not find a suitable product recommendation."},
            )

        top = docs[0]
        price_match = re.search(r"\$\s?(\d+(?:\.\d{2})?)", top.page_content)
        price = float(price_match.group(1)) if price_match else 0.0

        reason = f"it aligns with your interest in {fav_category}" if fav_category else "it matches your query"
        if last_purchase:
            reason += f" and is similar to your previous purchase '{last_purchase}'"

        recommendation = StructuredProductRecommendation(
            product_id=top.metadata.get("doc_id", "unknown"),
            name=top.metadata.get("doc_id", "Recommended Product").replace("_", " ").title(),
            description_snippet=top.page_content[:120],
            price=price,
            reason=reason,
        )
        return StructuredAgentResult(
            task_id=task.task_id, agent_name=self.name, status="success",
            result_data=recommendation,
        )

    @staticmethod
    def _extract_product_name(text: str) -> str:
        """Best-effort: take the first capitalised phrase or first sentence."""
        match = re.search(r"'([^']+)'|\"([^\"]+)\"", text)
        if match:
            return match.group(1) or match.group(2)
        return text.split(".")[0][:60]
