# services/orchestrator.py
from typing import Dict, Any, List, Optional
from common.models import (
    CustomerQuery, MaskedQuery, ChatbotResponse,
    RoutingRequest, AgentInvocation, AgentTask, StructuredAgentResult, NLGRequest
)
from services.pii_masker import PIIMasker
from services.llm_inference import MockLLMInferenceService
from services.ecommerce_client import ECommerceAPIClient
from services.agents.base_agent import BaseAgent # For type hinting the agents dictionary

class AgentOrchestratorService:
    """
    The central brain (CEO/Conductor) of the multi-agent system.
    Manages overall workflow, task decomposition, multi-agent coordination,
    conversation history, and final customer-facing natural language synthesis.
    """
    def __init__(self, llm_inference_client: MockLLMInferenceService, pii_masker: PIIMasker,
                 agents: Dict[str, BaseAgent], ecommerce_api_client: ECommerceAPIClient):
        self.llm_inference_client = llm_inference_client
        self.pii_masker = pii_masker
        self.agents = agents # Dictionary of agent_name -> Agent instance
        self.ecommerce_api_client = ecommerce_api_client
        self.conversation_history_db: Dict[str, List[Dict[str, str]]] = {} # Mock DB: {session_id: List[Dict]}
        self.agent_state_store: Dict[str, Dict[str, Any]] = {} # Mock Store: {session_id: Dict}
        self.orchestrator_routing_cache: Dict[str, AgentInvocation] = {} # Cache1

    def handle_customer_query(self, query: CustomerQuery) -> ChatbotResponse:
        print(f"\n--- Orchestrator: Handling Customer Query (Session: {query.session_id}, User: {query.user_id}) ---")
        
        # 1. PII Masking (Ideally done by Chatbot Interface/proxy before reaching Orchestrator)
        # For this skeleton, Orchestrator performs it upon reception
        masked_query = self.pii_masker.mask_text(query.text, session_id=query.session_id, user_id=query.user_id)
        print(f"  Masked Query: '{masked_query.masked_text}'")

        # 2. Conversation History Retrieval & Context (PII-masked)
        session_id = query.session_id
        user_id = query.user_id
        is_first_turn = session_id not in self.conversation_history_db
        current_history = self.conversation_history_db.get(session_id, [])
        current_history.append({"role": "user", "content": masked_query.masked_text})
        
        # 3. Initial Query Routing (LLMInf_Router)
        print("  Consulting Orchestrator Routing Cache (Cache1)...")
        routing_key = masked_query.masked_text # Simple key, could be (user_id, masked_query_text)
        agent_invocation: AgentInvocation
        if routing_key in self.orchestrator_routing_cache:
            agent_invocation = self.orchestrator_routing_cache[routing_key]
            print("  Cache hit for routing!")
        else:
            print("  Cache miss. Calling LLMInf_Router...")
            routing_request = RoutingRequest(
                session_id=session_id,
                conversation_history=current_history,
                current_query=masked_query.masked_text
            )
            agent_invocation = self.llm_inference_client.call_router(routing_request)
            self.orchestrator_routing_cache[routing_key] = agent_invocation # Cache the routing decision

        print(f"  Routing decision: Invoke agent '{agent_invocation.agent_name}' with confidence {agent_invocation.confidence}")

        # 4. Agent Task Assignment & Coordination
        target_agent: BaseAgent = self.agents.get(agent_invocation.agent_name) # Ensure type hinting for BaseAgent
        if not target_agent:
            print(f"  Warning: Agent '{agent_invocation.agent_name}' not found. Falling back to Escalation Agent.")
            target_agent = self.agents["EscalationAgent"]
            agent_invocation.agent_name = "EscalationAgent"
            agent_invocation.parameters = {"reason": f"No agent found for intent: {agent_invocation.agent_name}"}

        agent_task = AgentTask(
            session_id=session_id,
            customer_id=user_id, # Or a tokenized version derived from user_id
            original_query=masked_query.masked_text,
            intent=agent_invocation.agent_name, # Simplified intent for this demo
            params=agent_invocation.parameters,
            conversation_context=current_history
        )
        
        # Simulating parallel invocation (though in this single-threaded demo, it's sequential)
        agent_results: List[StructuredAgentResult] = []
        try:
            result = target_agent.process_task(agent_task)
            agent_results.append(result)
        except Exception as e:
            print(f"  Error processing task by {target_agent.name}: {e}. Falling back to Escalation Agent.")
            escalation_task = AgentTask(
                session_id=session_id,
                customer_id=user_id,
                original_query=masked_query.masked_text,
                intent="escalation_due_to_error",
                params={"reason": f"Agent {target_agent.name} failed with error: {e}"},
                conversation_context=current_history
            )
            agent_results.append(self.agents["EscalationAgent"].process_task(escalation_task))


        # 5. Final Natural Language Synthesis (LLMInf_Generative)
        print("\n  Aggregating agent results for final NLG (LLMInf_Generative)...")
        # Only look up the customer's name on their first turn in this session —
        # greeting by name on every single reply reads as robotic, and it saves
        # a DB round-trip on every message. Never derive a "name" from
        # session_id: that's a conversation correlator, not identity.
        customer_name = self.ecommerce_api_client.get_customer_name(user_id) if is_first_turn else None
        nlg_request = NLGRequest(
            session_id=session_id,
            conversation_history=current_history,
            agent_results=agent_results,
            final_user_intent=agent_invocation.agent_name, # Simplified
            customer_name=customer_name,
        )
        final_response_text = self.llm_inference_client.call_generative(nlg_request)
        
        # 6. Update Conversation History & State
        self.conversation_history_db[session_id] = current_history + [{"role": "assistant", "content": final_response_text}]
        self.agent_state_store[session_id] = {"last_agent": agent_invocation.agent_name, "last_result": agent_results}

        print("--- Orchestrator: Query Handled ---")
        return ChatbotResponse(
            session_id=session_id,
            response_text=final_response_text,
            agent_invoked=agent_invocation.agent_name
        )

# Example of how this service might be run (e.g., as a FastAPI endpoint)
if __name__ == "__main__":
    print("Orchestrator Service requires all dependencies to be initialized.")
    # This block is for conceptual understanding; full setup is in main_simulation.py