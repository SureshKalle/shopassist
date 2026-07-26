# services/agents/base_agent.py
"""
Shared contract and dependency set for every specialist agent
(services/agents/order_tracking_agent.py, product_recommendation_agent.py,
general_purpose_agent.py, escalation_agent.py). Each agent gets the same three
dependencies injected by api/dependencies.py / main_simulation.py, then
implements process_task() with its own domain logic - deciding whether to call
EcommerceClient, query the RAG store, or ask the LLM to reason/interpret.

PII masking isn't one of the three: it happens once, centrally, in
services/orchestrator.py before any agent ever runs (every agent already
receives PII-masked text) - no agent needs its own PIIMasker instance.
"""
from abc import ABC, abstractmethod
from common.models import AgentTask, StructuredAgentResult
from services.llm_inference import LLMInferenceService
from services.rag import RAGService
from clients.ecommerce_api_client import EcommerceClient

class BaseAgent(ABC):
    """
    Abstract Base Class for all Specialized AI Agents.
    Defines the common interface and shared dependencies.
    """
    def __init__(self, name: str, llm_inference_client: LLMInferenceService, rag_service: RAGService,
                 ecommerce_api_client: EcommerceClient):
        # `name` is a literal string each concrete agent hardcodes into its
        # own super().__init__() call (e.g. OrderTrackingAgent passes
        # "OrderTrackingAgent") - it must exactly match the key that same
        # agent is registered under in api/dependencies.py::get_agents()
        # (and main_simulation.py's equivalent dict for the CLI entry
        # point). Dispatch itself doesn't depend on this - services/
        # orchestrator.py looks agents up by the registry's dict key, not by
        # self.name - but self.name is what ends up in every
        # StructuredAgentResult.agent_name this agent returns, and
        # ultimately the customer-facing ChatbotResponse.agent_invoked, so a
        # mismatch here silently reports the wrong agent name everywhere
        # instead of erroring.
        self.name = name
        self.llm_inference_client = llm_inference_client
        self.rag_service = rag_service
        self.ecommerce_api_client = ecommerce_api_client

    @abstractmethod
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        """
        Processes an AgentTask and returns a structured result.
        Each concrete agent must implement its domain-specific logic here.
        """
        pass

# Example of how an agent might be instantiated (not run directly)
if __name__ == "__main__":
    # You cannot instantiate BaseAgent directly because it's abstract
    # This block is mainly for demonstrating imports/dependencies
    print("BaseAgent defines the common interface for specialized agents.")