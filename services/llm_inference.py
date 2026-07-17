# services/llm_inference.py
"""
The one place every LLM call in shopassist goes through. Wraps an OpenAI-
compatible client pointed at a local Ollama server (see OLLAMA_API_BASE_URL /
OLLAMA_*_MODEL in .env.example) behind five purpose-specific methods, one per
step of the pipeline in services/orchestrator.py and services/agents/*.py:

- call_router           - which specialist agent should handle this message?
- call_agent_reason     - should this agent call a tool (DB/RAG), or return now?
- call_agent_interpret  - given raw tool output, what's the structured diagnosis?
- call_generative       - turn an agent's structured result into a customer reply.
- call_embeddings       - turn text into a vector (used by services/rag.py).

Each method sends a system prompt describing the exact JSON schema the LLM must
return, then validates the response against the matching Pydantic model in
common/models.py. If the model is unavailable, times out, or returns invalid
JSON, every method degrades to a safe fallback value instead of raising -
callers never need to handle an LLM-specific exception themselves.
"""
import logging
import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

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
load_dotenv()

logger = logging.getLogger(__name__)


class LLMInferenceService:
    """
    The Centralized LLM Inference Service.
    Wraps actual LLM API calls and provides specialized endpoints.
    """
    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
        self.openai_client = OpenAI(
            base_url=f"{self.ollama_base_url}/v1", # OpenAI-compatible endpoint
            api_key="ollama" 
        )
        
        # Define LLM models to be used for each endpoint
        self.router_model = os.getenv("OLLAMA_ROUTER_MODEL", "llama3:8b-instruct") 
        self.agent_reason_model = os.getenv("OLLAMA_AGENT_REASON_MODEL", "llama3:8b-instruct")
        self.agent_interpret_model = os.getenv("OLLAMA_AGENT_INTERPRET_MODEL", "llama3:8b-instruct")
        self.generative_model = os.getenv("OLLAMA_GENERATIVE_MODEL", "llama3:8b-instruct")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text") 

    def call_router(self, request: RoutingRequest) -> AgentInvocation:
        """Decide which specialist agent should handle this query.

        Sends the (PII-masked) conversation history plus the current query and
        asks the LLM to pick one of the five known agent names with a confidence
        score. Falls back to GeneralPurposeAgent - at a lower confidence each
        time - if the LLM names an unknown agent, returns invalid JSON, or the
        call itself fails (network error, model not pulled, etc.).
        """
        logger.info("call_router: model=%s query=%r", self.router_model, request.current_query)
        
        # Prepare conversation history for the LLM
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": (
                "You are an expert routing agent for an e-commerce customer service chatbot. "
                "Your task is to analyze the user's current query and conversation history to determine "
                "which specialized agent should handle the request. "
                "You must respond with a JSON object containing three fields: " 
                "1. `agent_name`: The name of the agent to invoke. Choose from: "
                "'OrderTrackingAgent', 'ProductRecommendationAgent', 'ReturnsAgent', 'GeneralPurposeAgent', 'EscalationAgent'. "
                "2. `parameters`: A JSON object containing any key-value pairs relevant to the agent's task "
                "(e.g., {'order_id': '12345'} for OrderTrackingAgent, {'product_type': 'laptop'} for ProductRecommendationAgent). "
                "If no specific parameters are extracted, return an empty object {}. "
                "3. `confidence`: A float between 0.0 and 1.0 representing your confidence in this routing decision. " # <--- ADDED HERE
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
            # Make the actual API call to the LLM
            response = self.openai_client.chat.completions.create(
                model=self.router_model,
                messages=messages,
                response_format={"type": "json_object"}, # Instruct LLM to generate JSON
                temperature=0.0, # Keep temperature low for deterministic routing
                seed=42 # For reproducibility in testing/capstone
            )
            
            # Extract and parse the JSON response
            llm_output_str = response.choices[0].message.content
            logger.debug("call_router: raw output=%s", llm_output_str)

            # Validate LLM output against Pydantic model
            parsed_invocation = AgentInvocation.model_validate_json(llm_output_str)

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

    def call_agent_reason(self, request: LLMAgentReasonRequest) -> LLMAgentReasonResponse:
        """Ask the LLM to plan the agent's single next step.

        Given a task description, the agent's current state, and the tools it's
        allowed to use (`request.available_tools`), the LLM returns one of
        'call_api' / 'query_rag' / 'return_result' / 'escalate', plus which tool
        to call and with what parameters if applicable. Callers must compare
        `response.tool_name` against the exact string they advertised in
        `available_tools` - see services/agents/order_tracking_agent.py for the
        convention (`'ECommerceAPI.getOrderDetails'`, not just `'ECommerceAPI'`).
        Falls back to `action='return_result'` on any unknown action, invalid
        JSON, or call failure.
        """
        logger.info("call_agent_reason: model=%s agent=%s", self.agent_reason_model, request.agent_name)

        # Prepare the reasoning prompt for the LLM
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": (
                "You are a reasoning engine for a specialised customer-support AI agent. "
                "Given a task description, the current state, and a list of available tools, "
                "decide the SINGLE NEXT action the agent should take. "
                "You must respond with a JSON object containing exactly these fields: "
                "1. `action`: one of 'call_api', 'query_rag', 'return_result', 'escalate'. "
                "2. `tool_name`: the name of the tool to call if action is 'call_api' or 'query_rag' "
                "(must be one of the tools listed in `available_tools`), otherwise null. "
                "3. `tool_params`: a JSON object of parameters required for that tool call, otherwise null. "
                "4. `thought`: a brief chain-of-thought explanation for this decision. "
                "Use 'return_result' once enough information has been gathered to answer the task. "
                "Use 'escalate' only if the task cannot be resolved with the available tools. "
                "Always output a valid JSON object. Do NOT include any other text."
            )},
            {"role": "user", "content": (
                f"Agent: {request.agent_name}\n"
                f"Task: {request.task_description}\n"
                f"Current state: {request.current_state}\n"
                f"Available tools: {request.available_tools}"
            )}
        ]

        try:
            # Make the actual API call to the LLM
            response = self.openai_client.chat.completions.create(
                model=self.agent_reason_model,
                messages=messages,
                response_format={"type": "json_object"}, # Instruct LLM to generate JSON
                temperature=0.0, # Keep temperature low for deterministic reasoning
                seed=42 # For reproducibility in testing/capstone
            )

            # Extract and parse the JSON response
            llm_output_str = response.choices[0].message.content
            logger.debug("call_agent_reason: raw output=%s", llm_output_str)

            # Validate LLM output against Pydantic model
            parsed_response = LLMAgentReasonResponse.model_validate_json(llm_output_str)

            # Simple check for known actions, fallback if LLM invents one
            valid_actions = {"call_api", "query_rag", "return_result", "escalate"}
            if parsed_response.action not in valid_actions:
                logger.warning("call_agent_reason: LLM suggested unknown action '%s' - falling back to return_result", parsed_response.action)
                return LLMAgentReasonResponse(
                    action="return_result",
                    thought=f"Unknown action '{parsed_response.action}' from LLM; defaulting to return_result."
                )

            logger.info(
                "call_agent_reason: agent=%s action=%s tool_name=%s",
                request.agent_name, parsed_response.action, parsed_response.tool_name,
            )
            return parsed_response

        except ValidationError as e:
            logger.warning("call_agent_reason: LLM output invalid (%s) - defaulting to return_result", e)
            # Fallback for malformed LLM output
            return LLMAgentReasonResponse(
                action="return_result",
                thought=f"LLM agent-reason output parse error: {e}"
            )
        except Exception as e:
            logger.error("call_agent_reason: error calling agent-reason LLM: %s", e, exc_info=True)
            # General fallback for API errors, network issues, etc.
            return LLMAgentReasonResponse(
                action="return_result",
                thought=f"LLM agent-reason general error: {e}"
            )


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
            "call_agent_interpret: model=%s agent=%s goal=%s",
            self.agent_interpret_model, request.agent_name, request.interpretation_goal,
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
            # Make the actual API call to the LLM
            response = self.openai_client.chat.completions.create(
                model=self.agent_interpret_model,
                messages=messages,
                response_format={"type": "json_object"}, # Instruct LLM to generate JSON
                temperature=0.0, # Keep temperature low for deterministic interpretation
                seed=42 # For reproducibility
            )
            
            # Extract and parse the JSON response
            llm_output_str = response.choices[0].message.content
            logger.debug("call_agent_interpret: raw output=%s", llm_output_str)

            # Validate LLM output against the specific Pydantic model for interpretation
            # Assuming 'diagnose order issue' is the primary goal for now, which maps to OrderIssueAnalysis
            # If other goals were introduced, this might become a Union and require more complex validation.
            structured_interpretation_data = OrderIssueAnalysis.model_validate_json(llm_output_str)

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
    
    def call_agent_generate(self, request: LLMAgentReasonRequest) -> AgentGenerationOutput:
        """Generate a short natural-language snippet for a single agent-level task.

        Not currently called by any agent or by the orchestrator - every agent
        today returns a structured result and lets `call_generative()` do the
        one customer-facing synthesis step instead. Kept for an agent that
        needs its own standalone snippet (e.g. a one-off product blurb)
        without going through the full orchestrator NLG step.
        """
        logger.info("call_agent_generate: model=%s agent=%s", self.generative_model, request.agent_name)

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
            # Make the actual API call to the LLM
            response = self.openai_client.chat.completions.create(
                model=self.generative_model, # Using the general generative model for this
                messages=messages,
                response_format={"type": "json_object"}, # Instruct LLM to generate JSON
                temperature=0.7, # Higher temperature for more creative/varied generation
                seed=42 # For reproducibility
            )
            
            # Extract and parse the JSON response
            llm_output_str = response.choices[0].message.content
            logger.debug("call_agent_generate: raw output=%s", llm_output_str)

            # Validate LLM output against the Pydantic model
            parsed_generation = AgentGenerationOutput.model_validate_json(llm_output_str)

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
        
    def call_generative(self, request: NLGRequest) -> FinalNLGOutput:
        """Synthesize the final customer-facing reply from the agent's result(s).

        The last step of every request (see services/orchestrator.py step 5):
        formats whichever StructuredAgentResult the routed agent produced into
        plain text via `format_agent_results_for_llm()` below, then asks the LLM
        to write one coherent, on-tone response referencing it. Falls back to a
        generic apology string (`is_complete=False`) on invalid JSON or a call
        failure, so the customer always gets some reply.
        """
        logger.info("call_generative: model=%s intent=%s", self.generative_model, request.final_user_intent)

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
                        f"- Price: ${result_data.price:.2f}\n"
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
                    formatted_outputs.append(
                        f"### Escalation Notification ({status_indicator})\n"
                        f"- Reason: {result_data.escalation_reason}\n"
                        f"- Original Query: {result_data.original_query}\n"
                        f"- Conversation Summary Snippet: {result_data.conversation_summary[-1].content if result_data.conversation_summary else 'N/A'}"
                    )
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
            # Make the actual API call to the LLM
            response = self.openai_client.chat.completions.create(
                model=self.generative_model,
                messages=messages,
                response_format={"type": "json_object"}, # Instruct LLM to generate JSON
                temperature=0.7, # Higher temperature for more creative/natural generation
                seed=42 # For reproducibility
            )

            # Extract and parse the JSON response
            llm_output_str = response.choices[0].message.content
            logger.debug("call_generative: raw output=%s", llm_output_str)

            # Validate LLM output against Pydantic model
            parsed_nlg_output = FinalNLGOutput.model_validate_json(llm_output_str)

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

    def call_embeddings(self, text: str) -> List[float]:
        """Return a vector for `text`, used by services/rag.py for similarity search.

        This is a deterministic stub, not a real embedding model call: it maps
        the first 16 characters to floats via `ord()`. Good enough for the
        in-memory RAG demo (services/rag.py's MockRAGService does substring/
        keyword matching, not real vector similarity), not representative of
        real embedding quality or dimensionality (real models return
        hundreds-to-thousands of dimensions; this returns at most 16).
        """
        return [float(ord(c)) / 100 for c in text[:16]]

# Example of how this service might be run (e.g., as a FastAPI endpoint):
if __name__ == "__main__":
    # Set your OpenAI API key as an environment variable or uncomment and set it here
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" 
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not set. Using mock fallbacks for LLM calls.")
        # This fallback for demonstration if API key is not set, but won't be "real"
        class MockOpenAIClient:
            def chat(self):
                class MockCompletions:
                    def create(self, **kwargs):
                        class MockChoice:
                            message = type('obj', (object,), {'content': '{"agent_name": "GeneralPurposeAgent", "confidence": 0.5, "parameters": {}}'})()
                        return type('obj', (object,), {'choices': [MockChoice()]})()
                return MockCompletions()
            def embeddings(self):
                class MockEmbeddings:
                    def create(self, **kwargs):
                        class MockData:
                            embedding = [0.0] * 1536
                        return type('obj', (object,), {'data': [MockData()]})()
                return MockEmbeddings()
        LLMInferenceService.openai_client = MockOpenAIClient()
        
    llm_service = LLMInferenceService()
    
    # Mock a router call
    router_req = RoutingRequest(session_id="test_123", conversation_history=[], current_query="Check my order")
    agent_invoc = llm_service.call_router(router_req)
    print(f"\nRouter Result: {agent_invoc}")
    
    # Mock an embedding call
    embedding = llm_service.call_embeddings("Hello World")
    print(f"Embedding: {embedding[:5]}...")
