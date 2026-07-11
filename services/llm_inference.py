# services/llm_inference.py
import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from common.models import (
    RoutingRequest, AgentInvocation,
    LLMAgentReasonRequest, LLMAgentReasonResponse,
    LLMAgentInterpretRequest, LLMAgentInterpretResponse,
    NLGRequest, StructuredAgentResult,
    StructuredOrderSummary, StructuredProductRecommendation
)
load_dotenv() 
class MockLLMInferenceService:
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
        print(f"  [LLMInf] Calling LLMInf_Router ({self.router_model}) with query: '{request.current_query}'...")
        
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
            messages.append({"role": msg["role"], "content": msg["content"]})
        
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
            print(f"  [LLMInf] Router LLM raw output: {llm_output_str}")
            
            # Validate LLM output against Pydantic model
            parsed_invocation = AgentInvocation.model_validate_json(llm_output_str)
            
            # Simple check for known agents, fallback if LLM invents one
            if parsed_invocation.agent_name not in ["OrderTrackingAgent", "ProductRecommendationAgent", "ReturnsAgent", "GeneralPurposeAgent", "EscalationAgent"]:
                print(f"  [LLMInf] Warning: LLM suggested unknown agent '{parsed_invocation.agent_name}'. Falling back to GeneralPurposeAgent.")
                return AgentInvocation(agent_name="GeneralPurposeAgent", confidence=0.5, parameters={"original_query": request.current_query})
            
            return parsed_invocation

        except ValidationError as e:
            print(f"  [LLMInf] Error: LLM output for router is not valid JSON or doesn't match AgentInvocation schema: {e}")
            # Fallback for malformed LLM output
            return AgentInvocation(
                agent_name="GeneralPurposeAgent",
                confidence=0.3, # Lower confidence for fallback
                parameters={"original_query": request.current_query, "error": "LLM routing output parse error"}
            )
        except Exception as e:
            print(f"  [LLMInf] Error calling Router LLM: {e}")
            # General fallback for API errors, network issues, etc.
            return AgentInvocation(
                agent_name="GeneralPurposeAgent",
                confidence=0.2, # Very low confidence for general errors
                parameters={"original_query": request.current_query, "error": f"LLM routing general error: {e}"}
            )

    def call_agent_reason(self, request: LLMAgentReasonRequest) -> LLMAgentReasonResponse:
        
        print(f"  [LLMInf] Calling LLMInf_AgentReason ({self.agent_reason_model}) for {request.agent_name}...")

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
            print(f"  [LLMInf] AgentReason LLM raw output: {llm_output_str}")

            # Validate LLM output against Pydantic model
            parsed_response = LLMAgentReasonResponse.model_validate_json(llm_output_str)

            # Simple check for known actions, fallback if LLM invents one
            valid_actions = {"call_api", "query_rag", "return_result", "escalate"}
            if parsed_response.action not in valid_actions:
                print(f"  [LLMInf] Warning: LLM suggested unknown action '{parsed_response.action}'. Falling back to return_result.")
                return LLMAgentReasonResponse(
                    action="return_result",
                    thought=f"Unknown action '{parsed_response.action}' from LLM; defaulting to return_result."
                )

            return parsed_response

        except ValidationError as e:
            print(f"  [LLMInf] Error: LLM output for agent reason is not valid JSON or doesn't match LLMAgentReasonResponse schema: {e}")
            # Fallback for malformed LLM output
            return LLMAgentReasonResponse(
                action="return_result",
                thought=f"LLM agent-reason output parse error: {e}"
            )
        except Exception as e:
            print(f"  [LLMInf] Error calling AgentReason LLM: {e}")
            # General fallback for API errors, network issues, etc.
            return LLMAgentReasonResponse(
                action="return_result",
                thought=f"LLM agent-reason general error: {e}"
            )


    def call_agent_interpret(self, request: LLMAgentInterpretRequest) -> LLMAgentInterpretResponse:
        print(f"  [Mock LLMInf] Calling LLMInf_AgentInterpret for {request.agent_name} to {request.interpretation_goal}...")
        # Simulate interpretation of raw data
        if request.interpretation_goal == 'diagnose order issue' and request.raw_data.get('status') == 'Pending':
            return LLMAgentInterpretResponse(structured_interpretation={"issue_type": "PaymentPending", "recommendation": "Check payment method"}, thought="Interpreted order status as pending payment issue.")
        elif request.interpretation_goal == 'diagnose order issue' and request.raw_data.get('status') == 'Shipped':
             return LLMAgentInterpretResponse(structured_interpretation={"issue_type": "None", "recommendation": "Order is on its way"}, thought="Interpreted order status as shipped with no issues.")
        return LLMAgentInterpretResponse(structured_interpretation=request.raw_data, thought="Basic interpretation provided.")

    def call_agent_generate(self, request: LLMAgentReasonRequest) -> str: # Simplified for this example
        print(f"  [Mock LLMInf] Calling LLMInf_AgentGenerate for {request.agent_name}...")
        return "Generated snippet: This product is highly rated for durability."

    def call_generative(self, request: NLGRequest) -> str:
        print(f"  [Mock LLMInf] Calling LLMInf_Generative for final NLG...")
        # Simulate combining results into a natural language response
        responses = []
        for res in request.agent_results:
            if isinstance(res.result_data, StructuredOrderSummary):
                responses.append(f"Your order {res.result_data.order_id} is currently {res.result_data.status}.")
                if res.result_data.estimated_delivery:
                    responses.append(f"It's estimated to arrive by {res.result_data.estimated_delivery}.")
                if res.result_data.issue_analysis and res.result_data.issue_analysis != "None":
                    responses.append(f"Issue: {res.result_data.issue_analysis}")
            elif isinstance(res.result_data, StructuredProductRecommendation):
                responses.append(f"I recommend '{res.result_data.name}' (Price: ${res.result_data.price}) for you because {res.result_data.reason}.")
            elif isinstance(res.result_data, dict) and "answer_snippet" in res.result_data:
                responses.append(f"Here's some information: {res.result_data['answer_snippet']}")
            else:
                responses.append(f"Agent '{res.agent_name}' provided some information relevant to your query.")
        
        greeting = f"Hello {request.session_id.split('_')[0]}! " if request.session_id else "Hello! "
        return greeting + " ".join(responses) + f" Is there anything else I can assist you with regarding '{request.final_user_intent}'?"

    def call_embeddings(self, text: str) -> List[float]:
        # print(f"  [Mock LLMInf] Generating embedding for text snippet: '{text[:20]}...'")
        # Simulate embedding generation - simplified, actual embeddings are high-dimensional vectors
        return [float(ord(c)) / 100 for c in text[:16]] # Use first N chars to make mock embedding somewhat unique

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
                return MockEmbedAIClient()
        MockLLMInferenceService.openai_client = MockOpenAIClient()
        
    llm_service = LLMInferenceService()
    
    # Mock a router call
    router_req = RoutingRequest(session_id="test_123", conversation_history=[], current_query="Check my order")
    agent_invoc = llm_service.call_router(router_req)
    print(f"\nRouter Result: {agent_invoc}")
    
    # Mock an embedding call
    embedding = llm_service.call_embeddings("Hello World")
    print(f"Embedding: {embedding[:5]}...")
