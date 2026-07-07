# services/agents/general_purpose_agent.py
"""
GeneralPurposeAgent — LangChain tool-calling edition.

Single tool: `search_knowledge_base`, wrapping RAGService.as_retriever()
across all source types (policies, product info, past conversations).
The LLM decides whether/how many times to search before answering.
"""

import logging

from langchain_core.tools import tool

from common.models import AgentTask, StructuredAgentResult
from services.agents.base_agent import BaseAgent, EscalationSignal

logger = logging.getLogger(__name__)


class GeneralPurposeAgent(BaseAgent):
    """Handles general queries, FAQs, and store policy questions via RAG."""

    SYSTEM_PROMPT = (
        "You are GeneralPurposeAgent, a customer-support specialist for FAQs, "
        "store policies, and any question not handled by a more specific agent.\n\n"
        "You have a `search_knowledge_base` tool. Use it to find relevant policy "
        "or informational snippets, then answer the customer's question clearly "
        "and concisely, in your own words (do not just paste the raw snippet).\n\n"
        "If the knowledge base has nothing relevant, politely say you don't have "
        "that information and suggest rephrasing or contacting support directly. "
        "Do not escalate for simple 'I don't know' cases — only escalate if the "
        "customer seems distressed or the request is clearly outside a support bot's "
        "scope, using exactly: 'ESCALATE: <brief reason>'."
    )

    def __init__(self, *args, **kwargs):
        super().__init__("GeneralPurposeAgent", *args, **kwargs)

    # ------------------------------------------------------------------
    def _build_tools(self) -> list:
        retriever = self.rag_service.as_retriever(search_kwargs={"k": 3})

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the general knowledge base (policies, product info, past
            support conversations) for content relevant to the query."""
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant information found."
            return "\n---\n".join(d.page_content for d in docs)

        return [search_knowledge_base]

    # ------------------------------------------------------------------
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("[%s] Processing task %s", self.name, task.task_id)

        if self._agent_graph is None:
            return self._mock_process(task)

        instruction = f"Customer's question: '{task.original_query}'\n\nAnswer it helpfully."

        try:
            agent_result = self.run_agent(task, instruction)
        except EscalationSignal as esc:
            return StructuredAgentResult(
                task_id=task.task_id, agent_name=self.name, status="escalation",
                result_data={"escalation_reason": esc.reason},
            )
        except RuntimeError as exc:
            logger.warning("[%s] Agent graph unavailable (%s) — using mock.", self.name, exc)
            return self._mock_process(task)

        answer = agent_result["final_text"] or (
            f"I can help with questions like: '{task.original_query}'. "
            "Please rephrase or provide more details."
        )
        return StructuredAgentResult(
            task_id=task.task_id, agent_name=self.name, status="success",
            result_data={"answer_snippet": answer},
        )

    # ------------------------------------------------------------------
    def _mock_process(self, task: AgentTask) -> StructuredAgentResult:
        """Deterministic fallback path used when no LLM/API key is available."""
        retriever = self.rag_service.as_retriever(search_kwargs={"k": 1})
        docs = retriever.invoke(task.original_query)

        if docs:
            answer = docs[0].page_content
        else:
            answer = (
                f"I can help with questions like: '{task.original_query}'. "
                "Please rephrase or ask something else."
            )
        return StructuredAgentResult(
            task_id=task.task_id, agent_name=self.name, status="success",
            result_data={"answer_snippet": answer},
        )
