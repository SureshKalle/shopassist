# services/agents/base_agent.py
"""
Abstract Base Class for all Specialised AI Agents — LangChain edition.

Replaces the hand-written ReAct loop with LangChain's `create_agent`
(LangGraph-based tool-calling agent). The LLM itself now decides which tool
to call and when to stop — we no longer hand-roll the Reason→Act→Observe loop.

Design:
  - Each concrete agent defines its own tools as `@tool`-decorated closures
    over its dependencies (RAG service, e-commerce client) via `_build_tools()`.
  - `BaseAgent.__init__` compiles a LangGraph agent from those tools once.
  - `run_agent()` invokes the compiled graph with the task description and
    returns the final AI message plus the full tool-call trace (for the
    concrete agent to parse into a StructuredAgentResult).
  - Per-session conversational memory is handled via LangGraph's
    `InMemorySaver` checkpointer, keyed by `session_id` as the thread_id —
    this replaces the old `conversation_history_db` dict for agent-level
    (not orchestrator-level) short-term memory.

If no ANTHROPIC_API_KEY is configured, `run_agent()` raises RuntimeError
so callers can fall back to deterministic mock logic (kept in each
concrete agent's `_mock_process()`).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from clients.ecommerce_api_client import MockECommerceAPIClient
from common.llm_factory import build_chat_model
from common.models import AgentTask, StructuredAgentResult
from services.pii_masker import PIIMasker
from services.rag import RAGService

logger = logging.getLogger(__name__)


class EscalationSignal(Exception):
    """Raised when an agent (or the LLM within it) decides to escalate."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class BaseAgent(ABC):
    """
    Abstract base for all specialised customer-support agents, built on
    LangChain's tool-calling agent graph.

    Concrete subclasses must implement:
        _build_tools()  -> list of @tool-decorated callables
        process_task(task) -> StructuredAgentResult
    and should call `self.run_agent(task, task_description)` from within
    `process_task` to execute the LLM-driven tool loop.
    """

    #: Override in subclasses with a concise persona/instructions string.
    SYSTEM_PROMPT: str = "You are a helpful, precise customer-support agent."

    def __init__(
        self,
        name: str,
        llm_inference_client,          # kept for backward-compat call sites
        rag_service: RAGService,
        ecommerce_api_client: MockECommerceAPIClient,
        pii_masker: PIIMasker,
    ):
        self.name = name
        self.llm_inference_client = llm_inference_client
        self.rag_service = rag_service
        self.ecommerce_api_client = ecommerce_api_client
        self.pii_masker = pii_masker

        self._llm = self._build_llm()
        self._checkpointer = InMemorySaver()  # per-session short-term memory
        self._agent_graph = None
        if self._llm is not None:
            tools = self._build_tools()
            self._agent_graph = create_agent(
                model=self._llm,
                tools=tools,
                system_prompt=self.SYSTEM_PROMPT,
                checkpointer=self._checkpointer,
                name=self.name,
            )
            logger.info("[%s] LangChain agent graph compiled with %d tool(s).", self.name, len(tools))
        else:
            logger.warning(
                "[%s] No ANTHROPIC_API_KEY — agent graph not built; "
                "process_task() must use mock fallback logic.", self.name,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Abstract interface
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def _build_tools(self) -> list:
        """Return the list of @tool-decorated callables available to this agent."""

    @abstractmethod
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        """Process an AgentTask and return a structured result."""

    # ──────────────────────────────────────────────────────────────────────────
    # LLM / agent-graph plumbing
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_llm() -> Optional[BaseChatModel]:
        return build_chat_model(max_tokens=1024, temperature=0.2)

    def run_agent(self, task: AgentTask, task_instruction: str) -> dict[str, Any]:
        """
        Invoke the compiled LangGraph tool-calling agent.

        Args:
            task            : The AgentTask (used for session_id → thread_id mapping).
            task_instruction: A natural-language instruction describing what
                              the agent should accomplish this turn.

        Returns:
            {
              "final_text": str,                # the agent's final answer text
              "tool_calls": list[dict],          # {"tool": name, "input": ..., "output": ...}
              "messages": list[BaseMessage],     # full LangGraph message trace
            }

        Raises:
            RuntimeError if no LLM/agent graph is available (no API key) —
            callers should catch this and use their `_mock_process()` path.
            EscalationSignal if the agent's final answer signals escalation
            (looked for via a literal "ESCALATE:" prefix the system prompt
            instructs the model to use when it cannot help).
        """
        if self._agent_graph is None:
            raise RuntimeError(f"[{self.name}] Agent graph unavailable (no API key).")

        config: RunnableConfig = {"configurable": {"thread_id": task.session_id}}
        logger.info("[%s] Invoking agent graph | thread_id=%s", self.name, task.session_id)

        result = self._agent_graph.invoke(
            {"messages": [HumanMessage(content=task_instruction)]},
            config=config,
        )

        messages = result.get("messages", [])
        tool_calls: list[dict] = []
        final_text = ""

        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_calls.append(
                    {"tool": msg.name, "output": msg.content, "tool_call_id": msg.tool_call_id}
                )
            if isinstance(msg, AIMessage) and msg.content:
                final_text = msg.content if isinstance(msg.content, str) else str(msg.content)

        if final_text.strip().upper().startswith("ESCALATE:"):
            raise EscalationSignal(final_text.split(":", 1)[1].strip())

        logger.info(
            "[%s] Agent graph complete | %d tool call(s) | final_text=%.80s",
            self.name, len(tool_calls), final_text,
        )
        return {"final_text": final_text, "tool_calls": tool_calls, "messages": messages}

    # ──────────────────────────────────────────────────────────────────────────
    # Shared utility helpers (used by concrete agents' mock fallbacks & tools)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_order_id(text: str) -> Optional[str]:
        """Extract a numeric order ID from free text (e.g. 'order 12345', '#12345')."""
        import re
        match = re.search(r"(?:order\s*#?\s*|#)(\d{4,10})", text, re.IGNORECASE)
        return match.group(1) if match else None

    @staticmethod
    def _extract_product_query(text: str) -> str:
        """Strip filler phrases to get the core product query terms."""
        import re
        fillers = [
            r"\bcan you (recommend|suggest|help me (find|choose|pick))\b",
            r"\b(what|which) (is|are) (the )?(best|good|top)\b",
            r"\b(i('m| am) looking for|i need|i want)\b",
            r"[?.,!]",
        ]
        result = text.lower()
        for filler in fillers:
            result = re.sub(filler, " ", result, flags=re.IGNORECASE)
        stopwords = {"for", "please", "help", "me", "good", "great", "the", "a", "an", "my", "is", "around"}
        tokens = [t for t in result.split() if t not in stopwords]
        return " ".join(tokens).strip()
