# services/llm_inference.py
"""
The one place every LLM call in shopassist goes through. Wraps an OpenAI-
compatible client behind five purpose-specific methods, one per step of the
pipeline in services/orchestrator.py and services/agents/*.py:

- call_router           - which specialist agent should handle this message?
- call_agent_reason     - should this agent call a tool (DB/RAG), or return now?
- call_agent_interpret  - given raw tool output, what's the structured diagnosis?
- call_generative       - turn an agent's structured result into a customer reply.
- call_embeddings       - turn text into a vector (used by services/rag.py).

Each of the first four independently defaults to the local Ollama server
(OLLAMA_API_BASE_URL / OLLAMA_<ROLE>_MODEL in .env.example) and can be
switched to Gemini's OpenAI-compatible endpoint per-role via
<ROLE>_PROVIDER=gemini + GEMINI_API_KEY - see _resolve_role() below. This is
what lets a cheap/fast local model handle the frequent routing/reasoning
steps while an expensive cloud model is reserved for the one step that
actually reaches the customer (call_generative).

Each method sends a system prompt describing the exact JSON schema the LLM must
return, then validates the response against the matching Pydantic model in
common/models.py. If the model is unavailable, times out, or returns invalid
JSON, every method degrades to a safe fallback value instead of raising -
callers never need to handle an LLM-specific exception themselves.
"""
import logging
import os
from typing import List, Optional, Type, TypeVar
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
# --- NEW IMPORT ---
from openai import OpenAIError # For robust error handling with OpenAI client
# --- END NEW IMPORT ---
from pydantic import BaseModel, ValidationError

from common.models import (
    RoutingRequest, AgentInvocation,
    LLMAgentReasonRequest, LLMAgentReasonResponse,
    LLMAgentInterpretRequest, LLMAgentInterpretResponse,
    NLGRequest, StructuredAgentResult,
    StructuredOrderSummary,
    StructuredProductRecommendation,
    AgentGenerationOutput,
    GeneralPurposeAnswer,
    EscalationDetails,
    OrderIssueAnalysis,
    FinalNLGOutput
)
# --- Langfuse Integration Start ---
from langfuse import observe
# --- Langfuse Integration End ---

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# --- MODIFIED FALLBACK EMBEDDING DIMENSION ---
# Based on the traceback, nomic-embed-text generates 3072-dimensional embeddings.
FALLBACK_EMBEDDING = [0.0] * 3072
# --- END MODIFIED ---


class LLMInferenceService:
    """
    The Centralized LLM Inference Service.
    Wraps actual LLM API calls and provides specialized endpoints.
    """
    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
        self.local_client = OpenAI(
            base_url=f"{self.ollama_base_url}/v1", # OpenAI-compatible endpoint
            api_key="ollama"
        )
        # Lazily constructed by _gemini_client() - only touches GEMINI_API_KEY
        # if some role's *_PROVIDER is actually set to "gemini", so a fully
        # local deployment never needs cloud credentials to boot.
        self._gemini_client_singleton: Optional[OpenAI] = None

        # Per-role client/model/structured-output-mode/provider/local-fallback-
        # model. Each of these four roles independently defaults to "local"
        # (Ollama, same as before) and can be switched to Gemini via
        # <ROLE>_PROVIDER=gemini + GEMINI_API_KEY - see _resolve_role(). The 5th element
        # (local_model) is always resolved regardless of the active
        # provider, so a cloud-provider role can transparently retry against
        # Ollama if the cloud call fails (quota, outage, ...) - see
        # _complete_structured()'s fallback_model param. call_embeddings()
        # isn't listed here: it's a deterministic stub today (see its
        # docstring), not a real model call, so there's no client to route yet.
        self.router_client, self.router_model, self.router_mode, self.router_provider, self.router_local_model = self._resolve_role(
            "ROUTER", "llama3:8b-instruct", "gemini-2.5-flash"
        )
        self.agent_reason_client, self.agent_reason_model, self.agent_reason_mode, self.agent_reason_provider, self.agent_reason_local_model = self._resolve_role(
            "AGENT_REASON", "llama3:8b-instruct", "gemini-2.5-flash"
        )
        self.agent_interpret_client, self.agent_interpret_model, self.agent_interpret_mode, self.agent_interpret_provider, self.agent_interpret_local_model = self._resolve_role(
            "AGENT_INTERPRET", "llama3:8b-instruct", "gemini-2.5-flash"
        )
        # Shared by call_generative() and call_agent_generate() - both
        # already used the same self.generative_model before this refactor.
        # --- NOTE: If your local setup is specifically using "llama3.2:latest",
        #           you might need to change the default "llama3:8b-instruct" here
        #           to "llama3.2:latest" or ensure it's set via OLLAMA_GENERATIVE_MODEL env var.
        #           The traceback suggests it's trying to use llama3.2:latest for generative.
        self.generative_client, self.generative_model, self.generative_mode, self.generative_provider, self.generative_local_model = self._resolve_role(
            "GENERATIVE", "llama3:8b-instruct", "gemini-2.5-pro"
        )
        # --- END NOTE ---

        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

        # One line, always at INFO regardless of LOG_LEVEL, showing exactly
        # which provider/model handles each pipeline step for this process -
        # the fastest way to confirm a hybrid local/cloud config actually
        # took effect without digging through per-request logs.
        # --- MODIFIED LOGGING HERE ---
        logger.info(
            "LLMInferenceService ready | ROUTER=%s:%s AGENT_REASON=%s:%s AGENT_INTERPRET=%s:%s GENERATIVE=%s:%s EMBEDDING=ollama:%s",
            self.router_provider, self.router_model,
            self.agent_reason_provider, self.agent_reason_model,
            self.agent_interpret_provider, self.agent_interpret_model,
            self.generative_provider, self.generative_model,
            self.embedding_model # Added embedding model to startup log
        )
        # --- END MODIFIED LOGGING ---

    def _resolve_role(self, role: str, local_default_model: str, gemini_default_model: str) -> tuple[OpenAI, str, str, str, str]:
        """Pick the client/model/structured-output-mode/provider/local-model
        for one pipeline role.

        `<ROLE>_PROVIDER` (env var, default "local") chooses between:
        - "local": the existing Ollama server, model from OLLAMA_<ROLE>_MODEL.
        - "gemini": Gemini's OpenAI-compatible endpoint, model from
          GEMINI_<ROLE>_MODEL. Requires GEMINI_API_KEY.

        The returned mode ("json_object" or "parse") tells
        _complete_structured() which structured-output mechanism this
        provider actually speaks - see that method's docstring for why
        Gemini needs a different one than plain response_format=json_object.
        The returned provider string is only for logging (see __init__'s
        startup summary and each call_*'s entry log) - it's redundant with
        mode today (mode implies provider 1:1), but keeps the log lines
        reading "provider=gemini" instead of the reader having to remember
        that "mode=parse" means Gemini. The returned local_model is always
        computed (OLLAMA_<ROLE>_MODEL or its default) regardless of which
        provider is actually active, so callers always have a local fallback
        target on hand - see _complete_structured()'s fallback_model param.
        """
        local_model = os.getenv(f"OLLAMA_{role}_MODEL", local_default_model)
        provider = os.getenv(f"{role}_PROVIDER", "local").strip().lower()
        if provider == "local":
            return self.local_client, local_model, "json_object", provider, local_model
        if provider == "gemini":
            model = os.getenv(f"GEMINI_{role}_MODEL", gemini_default_model)
            return self._gemini_client(), model, "parse", provider, local_model
        raise ValueError(f"Unknown {role}_PROVIDER '{provider}' (expected 'local' or 'gemini')")

    def _gemini_client(self) -> OpenAI:
        if self._gemini_client_singleton is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is required when any *_PROVIDER env var is set to 'gemini' (see .env.example)."
                )
            self._gemini_client_singleton = OpenAI(
                api_key=api_key,
                base_url=os.getenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            )
        return self._gemini_client_singleton

    def _dispatch_structured(
        self, *, client: OpenAI, model: str, mode: str,
        messages: List[ChatCompletionMessageParam], schema: Type[T],
        temperature: float, seed: Optional[int] = None,
    ) -> T:
        """One single-attempt structured-output chat completion - no
        fallback, no logging beyond the raw response. See
        _complete_structured() (the only caller) for the two `mode`s this
        supports and why.
        """
        if mode == "parse":
            completion = client.beta.chat.completions.parse(
                model=model, messages=messages, temperature=temperature, response_format=schema,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError(f"{model} returned no parseable structured output")
            logger.debug("_dispatch_structured: model=%s parsed=%s", model, parsed.model_dump_json())
            return parsed

        kwargs = {"response_format": {"type": "json_object"}}
        if seed is not None:
            kwargs["seed"] = seed
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs,
        )
        llm_output_str = response.choices[0].message.content
        logger.debug("_dispatch_structured: model=%s raw output=%s", model, llm_output_str)
        return schema.model_validate_json(llm_output_str)

    def _complete_structured(
        self, *, caller: str, provider: str, client: OpenAI, model: str, mode: str,
        messages: List[ChatCompletionMessageParam], schema: Type[T],
        temperature: float, seed: Optional[int] = None,
        fallback_model: Optional[str] = None,
    ) -> T:
        """Run one structured-output chat completion, returning a validated
        `schema` instance regardless of which mechanism the provider speaks.

        - "json_object" (local Ollama, and most OpenAI-compatible hosts):
          plain chat.completions.create(response_format={"type": "json_object"}),
          manually validated against `schema` afterwards - the original
          behaviour of every call_* method before this existed.
        - "parse" (Gemini's OpenAI-compat endpoint): Gemini doesn't reliably
          honour response_format={"type": "json_object"} on
          chat.completions.create - Google's documented structured-output
          path is client.beta.chat.completions.parse(response_format=<a
          pydantic model>) instead, which returns an already-validated
          instance directly (https://ai.google.dev/gemini-api/docs/openai).

        `caller`/`provider` are logging-only (which call_* method, which
        provider actually handled it) - this is the one place every real LLM
        network call in the service passes through, so it's the definitive
        place to log "which model ran this request" at runtime.

        `fallback_model`: when set and `provider` isn't already "local", a
        failed primary call (cloud outage, quota exhausted, auth error,
        timeout, ...) is retried once against self.local_client with this
        model in "json_object" mode before giving up - transparent
        cloud->local failover, so a temporary cloud-side problem degrades to
        the free local model instead of straight to each call_* method's
        generic apology/fallback value. If the fallback attempt also fails,
        that exception propagates to the caller exactly as before this
        existed - callers' existing try/except still handles it.
        """
        logger.info("%s -> provider=%s model=%s mode=%s", caller, provider, model, mode)
        try:
            return self._dispatch_structured(
                client=client, model=model, mode=mode, messages=messages,
                schema=schema, temperature=temperature, seed=seed,
            )
        except Exception as e:
            if not fallback_model or provider == "local":
                raise
            logger.warning(
                "%s: %s call failed (%s) - falling back to local model %s",
                caller, provider, e, fallback_model,
            )
            return self._dispatch_structured(
                client=self.local_client, model=fallback_model, mode="json_object", messages=messages,
                schema=schema, temperature=temperature, seed=seed,
            )

    # --- Langfuse Integration Start: @observe decorator for call_router ---
    @observe(name="llm_inference_call_router")
    # --- Langfuse Integration End ---
    def call_router(self, request: RoutingRequest) -> AgentInvocation:
        """Decide which specialist agent should handle this query.

        Sends the (PII-masked) conversation history plus the current query and
        asks the LLM to pick one of the five known agent names with a confidence
        score. Falls back to GeneralPurposeAgent - at a lower confidence each
        time - if the LLM names an unknown agent, returns invalid JSON, or the
        call itself fails (network error, model not pulled, etc.).
        """
        logger.info("call_router: provider=%s model=%s query=%r", self.router_provider, self.router_model, request.current_query)

        # Prepare conversation history for the LLM
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": (
                "You are an expert routing agent for an e-commerce customer service chatbot. "
                "Your task is to analyze the user's current query and conversation history to determine "
                "which specialized agent should handle the request. "
                "You must respond with a JSON object containing three fields: "
                "1. `agent_name`: The name of the agent to invoke. Choose from: "
                "'OrderTrackingAgent', 'ProductRecommendationAgent', 'GeneralPurposeAgent', 'EscalationAgent'. "
                "2. `parameters`: A JSON object containing any key-value pairs relevant to the agent's task "
                "(e.g., {'order_id': '12345'} for OrderTrackingAgent, {'product_type': 'laptop'} for ProductRecommendationAgent). "
                "If no specific parameters are extracted, return an empty object {}. "
                "3. `confidence`: A float between 0.0 and 1.0 representing your confidence in this routing decision. "
                "If the intent is unclear or too broad for a specialized agent, default to 'GeneralPurposeAgent'. "
                "If the request implies an unresolvable issue or an explicit need for human intervention, choose 'EscalationAgent'. "
                "Always output a valid JSON object. Do NOT include any other text."
            )}
        ]

        # Add conversation history
        for msg in request.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current user query
        messages.append({"role": "user", "content": request.current_query})

        try:
            parsed_invocation = self._complete_structured(
                caller="call_router", provider=self.router_provider,
                client=self.router_client, model=self.router_model, mode=self.router_mode,
                messages=messages, schema=AgentInvocation,
                fallback_model=self.router_local_model,
                temperature=0.0, seed=42, # low temp + fixed seed for deterministic routing
            )

            # Simple check for known agents, fallback if LLM invents one
            if parsed_invocation.agent_name not in ["OrderTrackingAgent", "ProductRecommendationAgent", "ReturnsAgent", "GeneralPurposeAgent", "EscalationAgent"]:
                logger.warning("call_router: LLM suggested unknown agent '%s' - falling back to GeneralPurposeAgent", parsed_invocation.agent_name)
                return AgentInvocation(agent_name="GeneralPurposeAgent", confidence=0.5, parameters={"original_query": request.current_query})

            logger.info("call_router: agent=%s confidence=%s", parsed_invocation.agent_name, parsed_invocation.confidence)
            return parsed_invocation

        except ValidationError as e:
            logger.warning("call_router: LLM output invalid (%s) - falling back to GeneralPurposeAgent", e)
            # Fallback for malformed LLM output
            return AgentInvocation(
                agent_name="GeneralPurposeAgent",
                confidence=0.3, # Lower confidence for fallback
                parameters={"original_query": request.current_query, "error": "LLM routing output parse error"}
            )
        except Exception as e:
            logger.error("call_router: error calling router LLM: %s", e, exc_info=True)
            # General fallback for API errors, network issues, etc.
            return AgentInvocation(
                agent_name="GeneralPurposeAgent",
                confidence=0.2, # Very low confidence for general errors
                parameters={"original_query": request.current_query, "error": f"LLM routing general error: {e}"}
            )

    # --- Langfuse Integration Start: @observe decorator for call_agent_reason ---
    @observe(name="llm_inference_call_agent_reason")
    # --- Langfuse Integration End ---
    def call_agent_reason(self, request: LLMAgentReasonRequest) -> LLMAgentReasonResponse:
        logger.info("call_agent_reason: provider=%s model=%s agent=%s", self.agent_reason_provider, self.agent_reason_model, request.agent_name)

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": (
                "You are a reasoning engine for a specialised customer-support AI agent. "
                "Given a task description, the current state, and a list of available tools, "
                "decide the SINGLE NEXT action the agent should take. "
                "You must respond with a JSON object containing exactly these fields: "
                "1. `action`: one of 'call_api', 'query_rag', 'return_result', 'escalate'. "
                "2. `tool_name`: The general name of the tool category (e.g., 'ECommerceAPI', 'RAG'). "
                "   Return null if `action` is 'return_result' or 'escalate'. "
                "3. `tool_params`: A JSON object of parameters required for that tool call. "
                "   If `action` is 'call_api' or 'query_rag', this object MUST include a `method` field "
                "   specifying the exact function to call within that tool (e.g., {'method': 'getOrderDetails', 'order_id': '123'}). "
                "   Otherwise, return null. "
                "4. `thought`: A brief chain-of-thought explanation for this decision. "
                "Use 'return_result' once enough information has been gathered to answer the task. "
                "Use 'escalate' only if the task cannot be resolved with the available tools. "
                "Always output a valid JSON object. Do NOT include any other text."
            )},
            {"role": "user", "content": (
                f"Agent: {request.agent_name}\n"
                f"Task: {request.task_description}\n"
                f"Current state: {request.current_state}\n"
                f"Available tools: {request.available_tools}" # This list is still useful for context
            )}
        ]

        try:
            parsed_response = self._complete_structured(
                caller="call_agent_reason", provider=self.agent_reason_provider,
                client=self.agent_reason_client, model=self.agent_reason_model, mode=self.agent_reason_mode,
                messages=messages, schema=LLMAgentReasonResponse,
                fallback_model=self.agent_reason_local_model,
                temperature=0.0, seed=42,
            )

            valid_actions = {"call_api", "query_rag", "return_result", "escalate"}
            if parsed_response.action not in valid_actions:
                logger.warning("call_agent_reason: LLM suggested unknown action '%s' - falling back to return_result", parsed_response.action)
                return LLMAgentReasonResponse(
                    action="return_result",
                    thought=f"Unknown action '{parsed_response.action}' from LLM; defaulting to return_result."
                )

            # --- MODIFIED VALIDATION LOGIC ---
            if parsed_response.action in ["call_api", "query_rag"]:
                if not parsed_response.tool_name:
                    logger.warning("call_agent_reason: LLM suggested action '%s' but no tool_name - falling back to return_result", parsed_response.action)
                    return LLMAgentReasonResponse(action="return_result", thought="LLM suggested tool action without a tool_name.")

                if not parsed_response.tool_params or "method" not in parsed_response.tool_params:
                    logger.warning("call_agent_reason: LLM suggested action '%s' but missing 'method' in tool_params - falling back to return_result", parsed_response.action)
                    return LLMAgentReasonResponse(action="return_result", thought="LLM suggested tool action without 'method' in tool_params.")

                # Now, instead of checking if tool_name is in available_tools, we check if the full method name is implicitly valid
                # For now, we rely on the agent's 'tools' dict to do the final validation.
                # The LLM is now trained to put "ECommerceAPI" in tool_name and "getOrderDetails" in tool_params['method']
                full_tool_identifier = f"{parsed_response.tool_name}.{parsed_response.tool_params['method']}"
                if full_tool_identifier not in request.available_tools:
                    logger.warning(
                        "call_agent_reason: LLM suggested tool '%s' with method '%s' not in available tools %s - falling back to return_result",
                        parsed_response.tool_name, parsed_response.tool_params['method'], request.available_tools
                    )
                    return LLMAgentReasonResponse(
                        action="return_result",
                        thought=f"LLM suggested unknown/unavailable tool method '{full_tool_identifier}'; defaulting to return_result."
                    )
            # --- END MODIFIED VALIDATION LOGIC ---

            logger.info(
                "call_agent_reason: agent=%s action=%s tool_name=%s method=%s",
                request.agent_name, parsed_response.action, parsed_response.tool_name,
                parsed_response.tool_params.get('method') if parsed_response.tool_params else None,
            )
            return parsed_response

        except ValidationError as e:
            logger.warning("call_agent_reason: LLM output invalid (%s) - defaulting to return_result", e)
            return LLMAgentReasonResponse(
                action="return_result",
                thought=f"LLM agent-reason output parse error: {e}"
            )
        except Exception as e:
            logger.error("call_agent_reason: error calling agent-reason LLM: %s", e, exc_info=True)
            return LLMAgentReasonResponse(
                action="return_result",
                thought=f"LLM agent-reason general error: {e}"
            )

    # --- Langfuse Integration Start: @observe decorator for call_agent_interpret ---
    @observe(name="llm_inference_call_agent_interpret")
    # --- Langfuse Integration End ---
    def call_agent_interpret(self, request: LLMAgentInterpretRequest) -> LLMAgentInterpretResponse:
        """Turn a tool's raw output into a structured diagnosis.

        Only one `interpretation_goal` is implemented today - 'diagnose order
        issue', which always validates against `OrderIssueAnalysis`. A second
        goal (e.g. for ProductRecommendationAgent) would need this method to
        pick a different response model based on `request.interpretation_goal`,
        which it doesn't do yet. Falls back to `issue_type='Unknown'` with
        `severity='high'` on invalid JSON or a call failure.
        """
        logger.info(
            "call_agent_interpret: provider=%s model=%s agent=%s goal=%s",
            self.agent_interpret_provider, self.agent_interpret_model, request.agent_name, request.interpretation_goal,
        )

        # Prepare the interpretation prompt for the LLM
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": (
                "You are an expert AI assistant tasked with interpreting raw data "
                "and extracting structured insights based on a specific interpretation goal. "
                "You must respond with a JSON object representing the structured interpretation. "
                "For the 'diagnose order issue' goal, the JSON must contain: "
                "1. `issue_type`: A string describing the issue (e.g., 'PaymentPending', 'ShippingDelay', 'None', 'Unknown'). "
                "2. `recommendation`: A brief, actionable recommendation or summary related to the issue. "
                "3. `severity`: An optional string for severity ('low', 'medium', 'high'). "
                "4. `additional_notes`: Any optional extra context or details. "
                "If no issue is found, `issue_type` should be 'None' and `recommendation` should reflect no issue. "
                "Always output a valid JSON object strictly adhering to the schema. Do NOT include any other text."
            )},
            {"role": "user", "content": (
                f"Agent: {request.agent_name}\n"
                f"Interpretation Goal: {request.interpretation_goal}\n"
                f"Raw Data: {request.raw_data}"
            )}
        ]

        try:
            # Assuming 'diagnose order issue' is the primary goal for now, which maps to OrderIssueAnalysis
            # If other goals were introduced, this might become a Union and require more complex validation.
            structured_interpretation_data = self._complete_structured(
                caller="call_agent_interpret", provider=self.agent_interpret_provider,
                client=self.agent_interpret_client, model=self.agent_interpret_model, mode=self.agent_interpret_mode,
                messages=messages, schema=OrderIssueAnalysis,
                fallback_model=self.agent_interpret_local_model,
                temperature=0.0, seed=42, # low temp + fixed seed for deterministic interpretation
            )

            logger.info(
                "call_agent_interpret: agent=%s issue_type=%s",
                request.agent_name, structured_interpretation_data.issue_type,
            )
            return LLMAgentInterpretResponse(
                structured_interpretation=structured_interpretation_data,
                thought=f"Successfully interpreted raw data for goal: {request.interpretation_goal}"
            )

        except ValidationError as e:
            logger.warning("call_agent_interpret: LLM output invalid (%s)", e)
            # Fallback for malformed LLM output
            return LLMAgentInterpretResponse(
                structured_interpretation=OrderIssueAnalysis(issue_type="Unknown", recommendation=f"Failed to interpret data due to LLM output error: {e}", severity="high"),
                thought=f"LLM agent-interpret output parse error for goal '{request.interpretation_goal}'"
            )
        except Exception as e:
            logger.error("call_agent_interpret: error calling agent-interpret LLM: %s", e, exc_info=True)
            # General fallback for API errors, network issues, etc.
            return LLMAgentInterpretResponse(
                structured_interpretation=OrderIssueAnalysis(issue_type="Unknown", recommendation=f"An internal error occurred during interpretation: {e}", severity="high"),
                thought=f"LLM agent-interpret general error for goal '{request.interpretation_goal}'"
            )

    # --- Langfuse Integration Start: @observe decorator for call_agent_generate ---
    @observe(name="llm_inference_call_agent_generate")
    # --- Langfuse Integration End ---
    def call_agent_generate(self, request: LLMAgentReasonRequest) -> AgentGenerationOutput:
        """Generate a short natural-language snippet for a single agent-level task.

        Not currently called by any agent or by the orchestrator - every agent
        today returns a structured result and lets `call_generative()` do the
        one customer-facing synthesis step instead. Kept for an agent that
        needs its own standalone snippet (e.g. a one-off product blurb)
        without going through the full orchestrator NLG step.
        """
        logger.info("call_agent_generate: provider=%s model=%s agent=%s", self.generative_provider, self.generative_model, request.agent_name)

        # Prepare the generation prompt for the LLM. Uses task_description and
        # current_state from LLMAgentReasonRequest, assuming the agent has
        # already decided *what* to generate (e.g. "describe X product") via
        # its own call_agent_reason() step.
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": (
                "You are a concise natural language generation engine for a specialized AI agent. "
                "Your task is to generate a short, informative natural language snippet "
                "based on the provided task description and current state, suitable for a customer. "
                "You must respond with a JSON object containing two fields: "
                "1. `generated_text`: The natural language snippet. "
                "2. `confidence`: A float between 0.0 and 1.0 reflecting your certainty in the accuracy/relevance of the snippet. "
                "Do NOT include any introductory or concluding remarks beyond the JSON. "
                "Always output a valid JSON object. Do NOT include any other text."
            )},
            {"role": "user", "content": (
                f"Agent: {request.agent_name}\n"
                f"Generation Task: {request.task_description}\n" # The task determined by agent_reason
                f"Current State/Context: {request.current_state}" # Relevant data to base generation on
            )}
        ]

        try:
            parsed_generation = self._complete_structured(
                caller="call_agent_generate", provider=self.generative_provider,
                client=self.generative_client, model=self.generative_model, mode=self.generative_mode,
                messages=messages, schema=AgentGenerationOutput,
                fallback_model=self.generative_local_model,
                temperature=0.7, seed=42, # higher temp for creative/varied generation
            )

            logger.info("call_agent_generate: agent=%s confidence=%s", request.agent_name, parsed_generation.confidence)
            return parsed_generation

        except ValidationError as e:
            logger.warning("call_agent_generate: LLM output invalid (%s)", e)
            # Fallback for malformed LLM output
            return AgentGenerationOutput(
                generated_text="I encountered an issue while generating a response. Please try again or rephrase your query.",
                confidence=0.1, # Very low confidence for fallback
                context_used=[f"LLM output parse error: {e}"]
            )
        except Exception as e:
            logger.error("call_agent_generate: error calling agent-generate LLM: %s", e, exc_info=True)
            # General fallback for API errors, network issues, etc.
            return AgentGenerationOutput(
                generated_text="I apologize, an unexpected error prevented me from generating a specific response.",
                confidence=0.0, # Zero confidence for general errors
                context_used=[f"LLM general error: {e}"]
            )

    # --- Langfuse Integration Start: @observe decorator for call_generative ---
    @observe(name="llm_inference_call_generative")
    # --- Langfuse Integration End ---
    def call_generative(self, request: NLGRequest) -> FinalNLGOutput:
        """Synthesize the final customer-facing reply from the agent's result(s).

        The last step of every request (see services/orchestrator.py step 5):
        formats whichever StructuredAgentResult the routed agent produced into
        plain text via `format_agent_results_for_llm()` below, then asks the LLM
        to write one coherent, on-tone response referencing it. Falls back to a
        generic apology string (`is_complete=False`) on invalid JSON or a call
        failure, so the customer always gets some reply.
        """
        logger.info("call_generative: provider=%s model=%s intent=%s", self.generative_provider, self.generative_model, request.final_user_intent)

        # Helper to format structured agent results for the LLM
        def format_agent_results_for_llm(agent_results: List[StructuredAgentResult]) -> str:
            formatted_outputs = []
            for res in agent_results:
                result_data = res.result_data
                status_indicator = f"Status: {res.status.capitalize()}"

                if isinstance(result_data, StructuredOrderSummary):
                    items_str = ', '.join([item.get('name', 'item') for item in result_data.items])
                    issue_str = f"Issue Analysis: {result_data.issue_analysis}" if result_data.issue_analysis and result_data.issue_analysis != "None" else "No specific issue identified."
                    formatted_outputs.append(
                        f"### Order Tracking Result ({status_indicator})\n"
                        f"- Order ID: {result_data.order_id}\n"
                        f"- Current Status: {result_data.status}\n"
                        f"- Items: {items_str}\n"
                        f"- Estimated Delivery: {result_data.estimated_delivery or 'Not available'}\n"
                        f"- {issue_str}"
                    )
                elif isinstance(result_data, StructuredProductRecommendation):
                    formatted_outputs.append(
                        f"### Product Recommendation Result ({status_indicator})\n"
                        f"- Recommended Product: {result_data.name} (ID: {result_data.product_id})\n"
                        f"- Price: ₹{result_data.price:.2f}\n"
                        f"- Reason for Recommendation: {result_data.reason}\n"
                        f"- Description Snippet: {result_data.description_snippet}"
                    )
                elif isinstance(result_data, GeneralPurposeAnswer): # New handler for GeneralPurposeAgent
                    formatted_outputs.append(
                        f"### General Information Result ({status_indicator})\n"
                        f"- Answer Snippet: {result_data.answer_snippet}\n"
                        f"- Source Documents: {', '.join(result_data.source_documents_summary) if result_data.source_documents_summary else 'None'}"
                    )
                elif isinstance(result_data, EscalationDetails): # New handler for EscalationAgent
                    # --- FIX: Changed 'conversation_context' to 'conversation_summary' ---
                    formatted_outputs.append(
                        f"### Escalation Notification ({status_indicator})\n"
                        f"- Reason: {result_data.escalation_reason}\n"
                        f"- Original Query: {result_data.original_query}\n"
                        f"- Conversation Summary Snippet: {result_data.conversation_summary[-1].content if result_data.conversation_summary else 'N/A'}"
                    )
                    # --- END FIX ---
                elif isinstance(result_data, AgentGenerationOutput): # If an agent returned a raw generation
                     formatted_outputs.append(
                        f"### Agent Generated Snippet ({status_indicator})\n"
                        f"- Snippet: {result_data.generated_text}\n"
                        f"- Confidence: {result_data.confidence:.2f}"
                    )
                else: # Fallback for Dict[str, Any] or other unexpected types
                    formatted_outputs.append(
                        f"### Unstructured Agent Result from {res.agent_name} ({status_indicator})\n"
                        f"- Raw Data: {result_data}"
                    )
            return "\n\n" + "\n".join(formatted_outputs) if formatted_outputs else "No specific agent results were provided."

        # Sentiment-aware tone (services/orchestrator.py's step 1.5 /
        # services/classifier_client.py), folded into the base system
        # message's own content rather than appended as a second, trailing
        # system message. Tried the latter first - it reliably broke local
        # Ollama's json_object-mode output (a system-role message landing
        # after the user turns confused it into free-forming an unrelated
        # JSON shape, even though Gemini's schema-constrained .parse() mode
        # tolerated it fine) - one leading system message is safe for both.
        # Only added when a real classification came back - label="unknown"
        # means the classifier was unreachable, and leaving the base prompt
        # untouched in that case keeps behaviour identical to before this
        # existed when the classifier isn't running.
        sentiment = request.customer_sentiment
        sentiment_instruction = ""
        if sentiment and sentiment.label != "unknown":
            sentiment_instruction = (
                f" The customer's message sentiment was automatically classified as "
                f"'{sentiment.label}' (confidence {sentiment.score:.2f}). Calibrate your tone to "
                "it: for negative sentiment, lead with empathy and acknowledge their frustration "
                "before addressing the request; for positive sentiment, match their warmth; for "
                "neutral, respond simply and directly - don't overplay emotion that isn't there. "
                "This calibration must never change what information you report - only how you say it."
            )

        # Prepare conversation history for the LLM
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": (
                "You are the main customer service chatbot, designed to provide friendly, "
                "helpful, and accurate responses to e-commerce customers. "
                "Your goal is to synthesize information from various specialized agents and "
                "the ongoing conversation history into a single, coherent, natural language response. "
                "Prioritize direct answers from agent results. "
                "Maintain a helpful and polite tone. "
                "If an escalation is needed, clearly state that a human agent will be involved. "
                "You must respond with a JSON object containing three fields: "
                "1. `response_text`: The final natural language response to the customer. "
                "2. `tone`: The inferred tone of the response (e.g., 'helpful', 'empathetic', 'neutral'). "
                "3. `is_complete`: A boolean indicating if the customer's current query has been fully addressed (True/False). "
                "4. `confidence`: A float between 0.0 and 1.0 representing your confidence in the accuracy/completeness of this final response."
                "Always output a valid JSON object. Do NOT include any other text."
                + sentiment_instruction
            )}
        ]

        # Add conversation history
        for msg in request.conversation_history:
            messages.append({"role": msg.role, "content": msg.content}) # Use Message model attributes

        # Add agent results and user intent to the prompt
        formatted_results = format_agent_results_for_llm(request.agent_results)
        messages.append({"role": "user", "content": (
            f"Synthesize a response for the customer. Their primary intent was to '{request.final_user_intent}'.\n\n"
            f"Here are the structured results from the agents:\n{formatted_results}\n\n"
            "Please generate the final customer-facing response, maintaining the conversation flow and ensuring all relevant details from the agent results are included. If any agent result indicates an 'escalation', clearly state that a human agent will follow up."
        )})

        try:
            parsed_nlg_output = self._complete_structured(
                caller="call_generative", provider=self.generative_provider,
                client=self.generative_client, model=self.generative_model, mode=self.generative_mode,
                messages=messages, schema=FinalNLGOutput,
                fallback_model=self.generative_local_model,
                temperature=0.7, seed=42, # higher temp for creative/natural generation
            )

            logger.info("call_generative: confidence=%s is_complete=%s", parsed_nlg_output.confidence, parsed_nlg_output.is_complete)
            return parsed_nlg_output

        except ValidationError as e:
            logger.warning("call_generative: LLM output invalid (%s)", e)
            # Fallback for malformed LLM output
            return FinalNLGOutput(
                response_text="I apologize, I encountered an issue while formulating my response. Please try again or rephrase your query.",
                tone="apologetic",
                is_complete=False,
                confidence=0.1
            )
        except Exception as e:
            logger.error("call_generative: error calling generative LLM: %s", e, exc_info=True)
            # General fallback for API errors, network issues, etc.
            return FinalNLGOutput(
                response_text="I'm sorry, an unexpected error occurred. Please bear with me while I try to reconnect.",
                tone="apologetic",
                is_complete=False,
                confidence=0.0
            )

    # --- Langfuse Integration Start: @observe decorator ---
    #@observe(name="llm_inference_call_embeddings")
    # --- Langfuse Integration End ---
    def call_embeddings(self, text: str) -> List[float]:
        """Return a vector for `text`, used by services/rag.py for similarity search.

        This method now connects to the configured Ollama embedding model
        (e.g., 'nomic-embed-text') via the local_client's OpenAI-compatible
        embeddings endpoint.
        """
        if not text or not text.strip():
            logger.warning("call_embeddings: Received empty text, returning fallback embedding.")
            return FALLBACK_EMBEDDING

        logger.info("call_embeddings: model=%s text_len=%d", self.embedding_model, len(text))
        try:
            # The OpenAI client's embeddings.create method expects 'input' and 'model'
            response = self.local_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            embedding = response.data[0].embedding
            logger.debug("call_embeddings: successfully generated embedding of size %d", len(embedding))
            return embedding
        except OpenAIError as e:
            logger.error("call_embeddings: OpenAI API error for model %s: %s", self.embedding_model, e, exc_info=True)
            logger.warning("call_embeddings: Falling back to zero-vector embedding due to API error.")
            return FALLBACK_EMBEDDING
        except Exception as e:
            logger.error("call_embeddings: General error generating embedding for model %s: %s", self.embedding_model, e, exc_info=True)
            logger.warning("call_embeddings: Falling back to zero-vector embedding due to general error.")
            return FALLBACK_EMBEDDING

# Example of how this service might be run (e.g., as a FastAPI endpoint):
if __name__ == "__main__":
    # Local by default (needs Ollama running); set ROUTER_PROVIDER=gemini
    # (+ GEMINI_API_KEY) in the environment to exercise the cloud path instead.
    llm_service = LLMInferenceService()

    # Mock a router call
    router_req = RoutingRequest(session_id="test_123", conversation_history=[], current_query="Check my order")
    agent_invoc = llm_service.call_router(router_req)
    print(f"\nRouter Result: {agent_invoc}")

    # --- MODIFIED EMBEDDING TEST ---
    # Test a normal embedding call
    embedding = llm_service.call_embeddings("Hello World from the real embedding model!")
    print(f"Embedding (first 5 values): {embedding[:5]}... (Length: {len(embedding)})")

    # Test with empty text for fallback
    empty_embedding = llm_service.call_embeddings("")
    print(f"Empty Embedding (first 5 values): {empty_embedding[:5]}... (Length: {len(empty_embedding)})")
    # --- END MODIFIED EMBEDDING TEST ---