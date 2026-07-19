# services/orchestrator.py
"""
The single entry point for a customer message, shared by both `main_simulation.py`
and `api/routers/chat.py`. `AgentOrchestratorService.handle_customer_query()` is
the whole request lifecycle end to end: mask PII, route to a specialist agent,
run it, then turn its structured result into a natural-language reply. See the
numbered steps inline below, which match README.md's "Request flow" section.

All state here (`conversation_history_db`, `agent_state_store`,
`orchestrator_routing_cache`) is a plain in-process dict - it resets on restart
and isn't shared across multiple replicas. Fine for local dev/demo; would need
a real store (Redis, a DB table) before running more than one instance.
"""
import logging
from typing import Dict, Any, List
from common.models import (
    CustomerQuery, ChatbotResponse,
    RoutingRequest, AgentInvocation, AgentTask, StructuredAgentResult, NLGRequest,
    Message,
    FinalNLGOutput,
    SentimentResult,
)
from services.pii_masker import PIIMasker
from services.llm_inference import LLMInferenceService
from services.classifier_client import ClassifierClient
from services.guardrails import GuardrailService
from services.agents.base_agent import BaseAgent  # for type hinting the agents dict

logger = logging.getLogger(__name__)

class AgentOrchestratorService:
    """
    The central brain (CEO/Conductor) of the multi-agent system.
    Manages overall workflow, task decomposition, multi-agent coordination,
    conversation history, and final customer-facing natural language synthesis.
    """
    def __init__(self, llm_inference_client: LLMInferenceService, pii_masker: PIIMasker,
                 agents: Dict[str, BaseAgent], classifier_client: ClassifierClient = None,
                 guardrail_service: GuardrailService = None):
        self.llm_inference_client = llm_inference_client
        self.pii_masker = pii_masker
        self.agents = agents  # agent_name -> agent instance (see api/dependencies.py get_agents())
        # Defaults to a real client (pointed at the classifier service, off by
        # default until that's started separately - see ClassifierClient's
        # fail-soft behaviour) rather than requiring every caller to build one.
        self.classifier_client = classifier_client or ClassifierClient()
        # Stateless rule-based checks (services/guardrails.py) - defaults the
        # same way classifier_client does, so existing callers
        # (api/dependencies.py, main_simulation.py) need no changes to get
        # guardrails for free.
        self.guardrail_service = guardrail_service or GuardrailService()
        # Per-session conversation turns, oldest first. Not persisted anywhere.
        self.conversation_history_db: Dict[str, List[Message]] = {}
        # Last agent/result per session - currently write-only, no reader consults it yet.
        self.agent_state_store: Dict[str, Dict[str, Any]] = {}
        # Sentiment of each session's most recent customer message (step 1.5
        # below). Read by step 5 to calibrate the final reply's tone (see
        # NLGRequest.customer_sentiment / LLMInferenceService.call_generative)
        # - routing/agent selection still ignores it, only NLG consumes it.
        self.session_sentiment: Dict[str, SentimentResult] = {}
        # Routing decisions keyed by exact masked-text match, so a repeated identical
        # query skips a second LLM router call. Grows unbounded for the process lifetime.
        self.orchestrator_routing_cache: Dict[str, AgentInvocation] = {}

    def handle_customer_query(self, query: CustomerQuery) -> ChatbotResponse:
        """Run one customer message through the full pipeline and return a reply.

        Steps (mirrors README.md's "Request flow"):
          1. Mask PII in the incoming text.
          1.5. Classify sentiment of the message (ClassifierClient) - doesn't
               affect routing, only step 5's reply tone (see that step).
          2. Load/update this session's conversation history.
          3. Ask the LLM router which agent should handle it (cached by exact
             masked-text match).
          4. Run that agent; any exception falls back to EscalationAgent so a
             failure never surfaces as a raw error to the customer.
          5. Turn the agent's structured result into a natural-language reply,
             calibrated to the sentiment from step 1.5 when available.
          6. Persist the updated history and return the response.
        """
        logger.info("Handling customer query: session_id=%s user_id=%s", query.session_id, query.user_id)

        # 1. PII masking. Ideally this happens at the edge (a proxy/gateway) before
        # anything reaches the orchestrator; here the orchestrator does it itself
        # on receipt, since there's no separate edge layer in this project yet.
        masked_query = self.pii_masker.mask_text(query.text, session_id=query.session_id, user_id=query.user_id)
        logger.debug("Masked query: '%s'", masked_query.masked_text)

        # 1.5. Sentiment of the customer's message, via the (separate,
        # optional) encoder-model classifier service - see
        # services/classifier_client.py. Fails soft to "unknown" if that
        # service isn't running, so this never blocks a chat turn. Logged and
        # stashed on self.session_sentiment for now; not yet consulted by
        # routing or NLG (see that dict's comment in __init__).
        sentiment = self.classifier_client.classify_sentiment(masked_query.masked_text)
        self.session_sentiment[query.session_id] = sentiment
        logger.info(
            "Customer sentiment: session_id=%s label=%s stars=%d score=%.2f",
            query.session_id, sentiment.label, sentiment.stars, sentiment.score,
        )

        # 2. Load this session's history (PII-masked turns only - never raw text),
        # append the current turn, and hand the running list to every downstream
        # call (router, agent, NLG) so each has full conversational context.
        session_id = query.session_id

        current_history_raw = self.conversation_history_db.get(session_id, [])
        # Defensive: tolerate plain dicts here too, in case a caller ever seeds
        # conversation_history_db directly instead of through this method.
        current_history: List[Message] = [Message(**m) if isinstance(m, dict) else m for m in current_history_raw]
        current_history.append(Message(role="user", content=masked_query.masked_text))

        # 2.5. Input guardrail (services/guardrails.py) - screens for blatant
        # prompt-injection/jailbreak attempts before anything reaches the LLM
        # router. A block substitutes the routing decision below with
        # EscalationAgent instead of special-casing control flow - the same
        # agent-run -> NLG path every other route already takes, so the
        # customer still gets a natural reply, just never gets to the router
        # or a specialist agent.
        input_verdict = self.guardrail_service.screen_input(masked_query.masked_text)

        # 3. Route to a specialist agent. Cache hit skips a second LLM call for an
        # identical masked query (e.g. a user resending the same message).
        agent_invocation: AgentInvocation
        if input_verdict.blocked:
            logger.warning(
                "Routing bypassed - input guardrail blocked this message: session_id=%s category=%s",
                session_id, input_verdict.category,
            )
            agent_invocation = AgentInvocation(
                agent_name="EscalationAgent",
                confidence=1.0,
                parameters={"reason": f"Blocked by input guardrail (category={input_verdict.category})"},
            )
        else:
            logger.debug("Consulting orchestrator routing cache...")
            routing_key = masked_query.masked_text  # simple key; could incorporate user_id if needed
            if routing_key in self.orchestrator_routing_cache:
                agent_invocation = self.orchestrator_routing_cache[routing_key]
                logger.debug("Routing cache hit for session_id=%s", session_id)
            else:
                logger.debug("Routing cache miss - calling the LLM router")
                routing_request = RoutingRequest(
                    session_id=session_id,
                    conversation_history=current_history,
                    current_query=masked_query.masked_text
                )
                agent_invocation = self.llm_inference_client.call_router(routing_request)
                self.orchestrator_routing_cache[routing_key] = agent_invocation

        logger.info(
            "Routing decision: agent=%s confidence=%s",
            agent_invocation.agent_name, agent_invocation.confidence,
        )

        # 4. Run the chosen agent. If the router named an agent that isn't
        # registered (e.g. a name change on one side but not the other), or
        # confidence is too low to trust the routing decision, fall back instead
        # of erroring: low confidence goes to EscalationAgent (treat it as
        # unresolvable), anything else goes to GeneralPurposeAgent.
        target_agent: BaseAgent = self.agents.get(agent_invocation.agent_name)
        if not target_agent:
            logger.warning(
                "Agent '%s' not found - falling back to GeneralPurposeAgent/EscalationAgent",
                agent_invocation.agent_name,
            )
            if agent_invocation.confidence < 0.5:  # arbitrary cutoff, not tuned against real data
                target_agent = self.agents["EscalationAgent"]
                agent_invocation.agent_name = "EscalationAgent"
                agent_invocation.parameters = {"reason": f"No agent found for intent: {agent_invocation.agent_name}, low confidence."}
            else:
                target_agent = self.agents["GeneralPurposeAgent"]
                agent_invocation.agent_name = "GeneralPurposeAgent"
                agent_invocation.parameters = {"query": masked_query.masked_text}

        agent_task = AgentTask(
            session_id=session_id,
            user_id=query.user_id,
            original_query=masked_query.masked_text,
            intent=agent_invocation.agent_name,  # the agent name doubles as the intent label here
            params=agent_invocation.parameters,
            conversation_context=current_history
        )

        # Only one agent ever runs per request today (no fan-out/parallel agents),
        # so this list always ends up with exactly one result.
        agent_results: List[StructuredAgentResult] = []
        try:
            result = target_agent.process_task(agent_task)
            agent_results.append(result)
        except Exception as e:
            # Catch-all: whatever the agent raised (LLM error, bad tool response,
            # unhandled edge case), the customer still gets a coherent reply
            # instead of a stack trace - EscalationAgent packages it for a human.
            logger.error(
                "Agent %s failed processing task_id=%s: %s - falling back to EscalationAgent",
                target_agent.name, agent_task.task_id, e, exc_info=True,
            )
            escalation_task = AgentTask(
                session_id=session_id,
                user_id=query.user_id,
                original_query=masked_query.masked_text,
                intent="escalation_due_to_error",
                params={"reason": f"Agent {target_agent.name} failed with error: {e}"},
                conversation_context=current_history
            )
            agent_results.append(self.agents["EscalationAgent"].process_task(escalation_task))

        # 5. Turn the agent's structured result (a StructuredOrderSummary, a
        # StructuredProductRecommendation, etc.) into a natural-language reply.
        logger.debug("Aggregating agent results for final NLG...")
        nlg_request = NLGRequest(
            session_id=session_id,
            conversation_history=current_history,
            agent_results=agent_results,
            final_user_intent=agent_invocation.agent_name,  # simplified: the routed agent name stands in for intent
            customer_sentiment=self.session_sentiment.get(session_id),  # from step 1.5; None/"unknown" is a no-op in call_generative
        )

        final_nlg_output: FinalNLGOutput = self.llm_inference_client.call_generative(nlg_request)
        final_response_text = final_nlg_output.response_text

        # 5.5. Output guardrail (services/guardrails.py) - the counterpart to
        # step 1's input-side PII masking: scans what the LLM actually
        # generated (which can echo back a tool result or a shipping
        # address) for PII/secret-shaped substrings and redacts any hit
        # before it reaches the customer.
        output_verdict = self.guardrail_service.screen_output(final_response_text)
        final_response_text = output_verdict.safe_text

        # 6. Persist the turn and return. conversation_history_db/agent_state_store
        # are per-process only (see class docstring) - both reset on restart.
        self.conversation_history_db[session_id] = current_history + [Message(role="assistant", content=final_response_text)]
        self.agent_state_store[session_id] = {"last_agent": agent_invocation.agent_name, "last_result": agent_results}

        logger.info(
            "Query handled: session_id=%s agent=%s confidence=%s",
            session_id, agent_invocation.agent_name, final_nlg_output.confidence,
        )
        return ChatbotResponse(
            session_id=session_id,
            response_text=final_response_text,
            agent_invoked=agent_invocation.agent_name,
            confidence_score=final_nlg_output.confidence
        )

# Not meant to be run directly - api/dependencies.py and main_simulation.py both
# construct this with real dependencies and call handle_customer_query() on it.
if __name__ == "__main__":
    print("Orchestrator Service requires all dependencies to be initialized.")
    # This block is for conceptual understanding; full setup is in main_simulation