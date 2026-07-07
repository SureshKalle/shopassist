# services/llm_inference.py
"""
Centralised LLM Inference Service — LangChain edition.

Every specialised endpoint is now an LCEL chain:

    prompt (ChatPromptTemplate)  →  llm (ChatAnthropic)  →  parser (structured Pydantic output)

Benefits over the raw-API version:
  - Prompts are versioned, reusable ChatPromptTemplate objects (easy to swap/A-B test).
  - `.with_structured_output(PydanticModel)` replaces hand-rolled JSON parsing —
    LangChain handles tool-calling-based structured extraction and validation.
  - Chains are composable: `prompt | llm | parser` and can be traced/logged via
    LangSmith simply by setting LANGCHAIN_TRACING_V2=true in .env (no code change).
  - Swapping the model provider (Anthropic → OpenAI → Bedrock) only touches the
    `_build_llm()` factory below.
"""

import logging
from typing import Optional, cast

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common.llm_factory import build_chat_model
from common.config import (
    LLM_ROUTER_MAX_TOKENS,
    LLM_GENERATIVE_MAX_TOKENS,
    LLM_AGENT_REASON_MAX_TOKENS,
    LLM_AGENT_INTERPRET_MAX_TOKENS,
)
from common.models import (
    AgentInvocation,
    LLMAgentInterpretRequest,
    LLMAgentInterpretResponse,
    LLMAgentReasonRequest,
    LLMAgentReasonResponse,
    NLGRequest,
    RoutingRequest,
    StructuredOrderSummary,
    StructuredProductRecommendation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas used purely as LangChain structured-output targets.
# ---------------------------------------------------------------------------
class _RouterOutput(BaseModel):
    agent_name: str = Field(
        description=(
            "One of: OrderTrackingAgent, ProductRecommendationAgent, "
            "ReturnsAgent, GeneralPurposeAgent"
        )
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    order_id: Optional[str] = Field(default=None, description="Order ID if detected in the query")
    product_query: Optional[str] = Field(default=None, description="Product search terms if relevant")
    reason: str = Field(description="Brief reasoning for this routing decision")


class _AgentReasonOutput(BaseModel):
    action: str = Field(description="One of: call_api, query_rag, return_result, escalate")
    tool_name: Optional[str] = Field(default=None, description="Name of the tool to call, if any")
    tool_params: Optional[dict] = Field(default=None, description="Parameters for the tool call")
    thought: str = Field(description="Brief chain-of-thought reasoning")


class _AgentInterpretOutput(BaseModel):
    issue_type: Optional[str] = Field(
        default=None, description="None | PaymentPending | Delayed | Lost | Damaged | Other"
    )
    recommendation: Optional[str] = Field(default=None, description="Actionable advice")
    sentiment: Optional[str] = Field(default=None, description="positive | neutral | negative")
    answer: Optional[str] = Field(default=None, description="Direct answer, for FAQ/general goals")
    eligible: Optional[bool] = Field(default=None, description="Eligibility flag, for returns goals")
    product_id: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    price: Optional[float] = Field(default=None)
    description_snippet: Optional[str] = Field(default=None)
    reason: Optional[str] = Field(default=None)
    thought: str = Field(description="Brief reasoning")


# ---------------------------------------------------------------------------
# LLM factory — delegates to common/llm_factory.py (provider-agnostic:
# Groq / Ollama / Anthropic, selected via LLM_PROVIDER in .env)
# ---------------------------------------------------------------------------
def _build_llm(max_tokens: int):
    return build_chat_model(max_tokens=max_tokens, temperature=0.2)


class LLMInferenceService:
    """
    LangChain-powered LLM Inference Service.

    Each `call_*` method uses an LCEL chain:
        ChatPromptTemplate | ChatAnthropic.with_structured_output(Schema)
    If no API key is configured, falls back to lightweight mock logic so the
    rest of the pipeline remains runnable offline.
    """

    # ── Router ────────────────────────────────────────────────────────────────

    _ROUTER_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert customer-support router for an e-commerce platform. "
                "Read the customer's latest message and recent conversation history, then "
                "decide which specialised agent should handle it.\n\n"
                "Available agents:\n"
                "  - OrderTrackingAgent        : order status, shipping, delivery updates\n"
                "  - ProductRecommendationAgent: product suggestions, comparisons, gift ideas\n"
                "  - ReturnsAgent              : returns, refunds, exchanges, cancellations\n"
                "  - GeneralPurposeAgent       : FAQs, store policies, anything else",
            ),
            ("human", "Conversation so far:\n{history}\n\nLatest customer message: {query}"),
        ]
    )

    def __init__(self):
        self._router_llm = _build_llm(LLM_ROUTER_MAX_TOKENS)
        self._reason_llm = _build_llm(LLM_AGENT_REASON_MAX_TOKENS)
        self._interpret_llm = _build_llm(LLM_AGENT_INTERPRET_MAX_TOKENS)
        self._generative_llm = _build_llm(LLM_GENERATIVE_MAX_TOKENS)

        self._router_chain = (
            self._ROUTER_PROMPT | self._router_llm.with_structured_output(_RouterOutput)
            if self._router_llm
            else None
        )

    def call_router(self, request: RoutingRequest) -> AgentInvocation:
        """Route the customer query to the best agent via an LCEL chain."""
        logger.info("[LLMInference] call_router | session=%s", request.session_id)

        if self._router_chain is None:
            return self._mock_router(request)

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in request.conversation_history[-6:]
        )
        try:
            result = cast(_RouterOutput, self._router_chain.invoke(
                {"history": history_text, "query": request.current_query}
            ))
            params = {"reason": result.reason}
            if result.order_id:
                params["order_id"] = result.order_id
            if result.product_query:
                params["product_query"] = result.product_query
            params.setdefault("query", request.current_query)
            return AgentInvocation(
                agent_name=result.agent_name,
                confidence=result.confidence,
                parameters=params,
            )
        except Exception as exc:
            logger.warning("[LLMInference] call_router chain failed (%s) — using mock.", exc)
            return self._mock_router(request)

    def _mock_router(self, request: RoutingRequest) -> AgentInvocation:
        q = request.current_query.lower()
        if any(k in q for k in ("order", "where is", "shipped", "delivery", "tracking")):
            return AgentInvocation(agent_name="OrderTrackingAgent", confidence=0.9,
                                   parameters={"query": request.current_query})
        if any(k in q for k in ("recommend", "suggest", "buy", "laptop", "product", "gift")):
            return AgentInvocation(agent_name="ProductRecommendationAgent", confidence=0.85,
                                   parameters={"query": request.current_query})
        if any(k in q for k in ("return", "refund", "exchange", "cancel")):
            return AgentInvocation(agent_name="ReturnsAgent", confidence=0.92,
                                   parameters={"query": request.current_query})
        return AgentInvocation(agent_name="GeneralPurposeAgent", confidence=0.6,
                               parameters={"query": request.current_query})

    # ── Agent Reason (legacy single-step planning; primary path for agents is
    #     now the LangChain tool-calling agent in base_agent.py) ────────────────

    _REASON_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a reasoning engine for a specialised customer-support AI agent. "
                "Given a task description, current state, and available tools, decide the "
                "SINGLE NEXT action.\n\n"
                "Actions:\n"
                "  call_api     : call an external API\n"
                "  query_rag    : query the knowledge base\n"
                "  return_result: enough info collected, compose the final answer\n"
                "  escalate     : cannot resolve, escalate to human agent",
            ),
            (
                "human",
                "Agent: {agent_name}\nTask: {task_description}\n"
                "Current state: {current_state}\nAvailable tools: {available_tools}",
            ),
        ]
    )

    def call_agent_reason(self, request: LLMAgentReasonRequest) -> LLMAgentReasonResponse:
        """
        Legacy single-step planning call (kept for any code path that doesn't
        use the LangChain tool-calling agent directly).
        """
        logger.info("[LLMInference] call_agent_reason | agent=%s", request.agent_name)

        if self._reason_llm is None:
            return self._mock_reason(request)

        try:
            chain = self._REASON_PROMPT | self._reason_llm.with_structured_output(_AgentReasonOutput)
            result = cast(_AgentReasonOutput, chain.invoke(
                {
                    "agent_name": request.agent_name,
                    "task_description": request.task_description,
                    "current_state": request.current_state,
                    "available_tools": request.available_tools,
                }
            ))
            return LLMAgentReasonResponse(
                action=result.action,
                tool_name=result.tool_name,
                tool_params=result.tool_params,
                thought=result.thought,
            )
        except Exception as exc:
            logger.warning("[LLMInference] call_agent_reason chain failed (%s) — mock.", exc)
            return self._mock_reason(request)

    def _mock_reason(self, request: LLMAgentReasonRequest) -> LLMAgentReasonResponse:
        desc = request.task_description.lower()
        if "order" in desc and "ECommerceAPI.getOrderDetails" in request.available_tools:
            order_id = request.current_state.get("order_id", "12345")
            return LLMAgentReasonResponse(
                action="call_api", tool_name="ECommerceAPI.getOrderDetails",
                tool_params={"order_id": order_id},
                thought=f"Need order details from API for {order_id}",
            )
        if "policy" in desc and "RAG.queryPolicy" in request.available_tools:
            return LLMAgentReasonResponse(
                action="query_rag", tool_name="RAG.queryPolicy",
                tool_params={"topic": "return_policy"},
                thought="Need to check return policy via RAG",
            )
        return LLMAgentReasonResponse(action="return_result", thought="No further tools needed.")

    # ── Agent Interpret ────────────────────────────────────────────────────────

    _INTERPRET_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a data interpretation engine for a customer-support AI agent. "
                "You receive raw data and an interpretation goal. Extract structured, "
                "actionable insight relevant to that goal. Only populate the fields that "
                "are relevant; leave others null.",
            ),
            (
                "human",
                "Agent: {agent_name}\nInterpretation goal: {goal}\nRaw data:\n{raw_data}",
            ),
        ]
    )

    def call_agent_interpret(
        self, request: LLMAgentInterpretRequest
    ) -> LLMAgentInterpretResponse:
        """Interpret raw API/tool data into structured insights via LCEL chain."""
        logger.info(
            "[LLMInference] call_agent_interpret | agent=%s | goal=%.50s",
            request.agent_name, request.interpretation_goal,
        )

        if self._interpret_llm is None:
            return self._mock_interpret(request)

        try:
            chain = self._INTERPRET_PROMPT | self._interpret_llm.with_structured_output(
                _AgentInterpretOutput
            )
            result = cast(_AgentInterpretOutput, chain.invoke(
                {
                    "agent_name": request.agent_name,
                    "goal": request.interpretation_goal,
                    "raw_data": request.raw_data,
                }
            ))
            data = result.model_dump(exclude_none=True, exclude={"thought"})
            return LLMAgentInterpretResponse(structured_interpretation=data, thought=result.thought)
        except Exception as exc:
            logger.warning("[LLMInference] call_agent_interpret chain failed (%s) — mock.", exc)
            return self._mock_interpret(request)

    def _mock_interpret(
        self, request: LLMAgentInterpretRequest
    ) -> LLMAgentInterpretResponse:
        status = request.raw_data.get("status", "")
        if status == "Pending":
            return LLMAgentInterpretResponse(
                structured_interpretation={
                    "issue_type": "PaymentPending",
                    "recommendation": "Check payment method",
                    "sentiment": "negative",
                },
                thought="Order pending — likely payment issue.",
            )
        if status == "Shipped":
            return LLMAgentInterpretResponse(
                structured_interpretation={
                    "issue_type": "None",
                    "recommendation": "Order is on its way",
                    "sentiment": "positive",
                },
                thought="Order shipped, no issues.",
            )
        return LLMAgentInterpretResponse(
            structured_interpretation=request.raw_data, thought="Basic interpretation."
        )

    # ── Generative NLG ────────────────────────────────────────────────────────

    _NLG_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly, professional customer-support representative for an "
                "e-commerce company. Craft a helpful, concise, empathetic response based on "
                "the structured results from our internal agents.\n\n"
                "Guidelines:\n"
                "- Be warm, clear, actionable.\n"
                "- Never expose internal system/agent/field names.\n"
                "- If unresolved/escalated, apologise and explain next steps.\n"
                "- Keep under 150 words.\n"
                "- End with an open offer to help further.",
            ),
            (
                "human",
                "Conversation history (recent):\n{history}\n\n"
                "Customer intent: {intent}\n\nAgent findings:\n{findings}",
            ),
        ]
    )

    def call_generative(self, request: NLGRequest) -> str:
        """Generate the final natural-language response via an LCEL chain."""
        logger.info("[LLMInference] call_generative | session=%s", request.session_id)

        findings = self._summarise_agent_results(request)
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in request.conversation_history[-4:]
        )

        if self._generative_llm is None:
            return self._mock_generative(request)

        try:
            chain = self._NLG_PROMPT | self._generative_llm | StrOutputParser()
            return chain.invoke(
                {
                    "history": history_text,
                    "intent": request.final_user_intent,
                    "findings": findings,
                }
            )
        except Exception as exc:
            logger.warning("[LLMInference] call_generative chain failed (%s) — mock.", exc)
            return self._mock_generative(request)

    @staticmethod
    def _summarise_agent_results(request: NLGRequest) -> str:
        lines = []
        for res in request.agent_results:
            if isinstance(res.result_data, StructuredOrderSummary):
                s = res.result_data
                lines.append(
                    f"Order {s.order_id}: status={s.status}, "
                    f"delivery={s.estimated_delivery or 'TBD'}, issue={s.issue_analysis or 'None'}"
                )
            elif isinstance(res.result_data, StructuredProductRecommendation):
                r = res.result_data
                lines.append(f"Recommended product '{r.name}' at ${r.price} — {r.reason}")
            else:
                lines.append(f"[{res.agent_name} | {res.status}] {res.result_data}")
        return "\n".join(lines)

    def _mock_generative(self, request: NLGRequest) -> str:
        parts = []
        for res in request.agent_results:
            if isinstance(res.result_data, StructuredOrderSummary):
                s = res.result_data
                parts.append(f"Your order {s.order_id} is currently {s.status}.")
                if s.estimated_delivery:
                    parts.append(f"Estimated delivery: {s.estimated_delivery}.")
                if s.issue_analysis and s.issue_analysis not in ("None", None):
                    parts.append(f"Note: {s.issue_analysis}.")
            elif isinstance(res.result_data, StructuredProductRecommendation):
                r = res.result_data
                parts.append(f"I recommend '{r.name}' at ${r.price} because {r.reason}.")
            elif isinstance(res.result_data, dict):
                snippet = res.result_data.get("answer_snippet") or res.result_data.get(
                    "escalation_reason", ""
                )
                if snippet:
                    parts.append(snippet)
        parts.append("Is there anything else I can help you with?")
        return " ".join(parts)

    # ── Embeddings ────────────────────────────────────────────────────────────

    def call_embeddings(self, text: str) -> list[float]:
        """
        Legacy direct entry point — kept for backward compatibility.
        New code should use `services.rag.get_embeddings_model()` directly
        (it implements LangChain's `Embeddings` interface and is what the
        `Chroma` vectorstore uses internally).
        """
        from services.rag import get_embeddings_model

        return get_embeddings_model().embed_query(text)


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------
MockLLMInferenceService = LLMInferenceService


if __name__ == "__main__":
    import uuid

    logging.basicConfig(level=logging.INFO)
    svc = LLMInferenceService()

    req = RoutingRequest(
        session_id=str(uuid.uuid4()), conversation_history=[],
        current_query="Where is my order 12345?",
    )
    print(f"Router → {svc.call_router(req)}")
    print(f"Embedding dim={len(svc.call_embeddings('hello'))}")
