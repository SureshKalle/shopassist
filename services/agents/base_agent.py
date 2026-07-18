# services/agents/base_agent.py
"""
Shared contract and dependency set for every specialist agent
(services/agents/order_tracking_agent.py, product_recommendation_agent.py,
general_purpose_agent.py, escalation_agent.py). Each agent gets the same four
dependencies injected by api/dependencies.py / main_simulation.py, then
implements process_task() with its own domain logic - deciding whether to call
EcommerceClient, query the RAG store, or ask the LLM to reason/interpret.
"""
from abc import ABC, abstractmethod
from common.models import AgentTask, StructuredAgentResult
from services.llm_inference import LLMInferenceService
from services.rag import MockRAGService
from clients.ecommerce_api_client import EcommerceClient
from services.pii_masker import PIIMasker

class BaseAgent(ABC):
    """
    Abstract Base Class for all Specialized AI Agents.
    Defines the common interface and shared dependencies.
    """
    def __init__(self, name: str, llm_inference_client: LLMInferenceService, rag_service: MockRAGService,
                 ecommerce_api_client: EcommerceClient, pii_masker: PIIMasker):
        self.name = name
        self.llm_inference_client = llm_inference_client
        self.rag_service = rag_service
        self.ecommerce_api_client = ecommerce_api_client
        self.pii_masker = pii_masker
        # Reserved for caching an agent's own reasoning decisions; declared but
        # not yet read or written by any agent - available for a subclass that
        # wants to cache its process_task() logic the way orchestrator.py
        # caches routing decisions.
        self.agent_reasoning_cache = {}

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