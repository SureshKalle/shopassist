# services/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from common.models import AgentTask, StructuredAgentResult
from services.llm_inference import MockLLMInferenceService
from services.rag import MockRAGService
from services.ecommerce_client import ECommerceAPIClient
from services.pii_masker import PIIMasker

class BaseAgent(ABC):
    """
    Abstract Base Class for all Specialized AI Agents.
    Defines the common interface and shared dependencies.
    """
    def __init__(self, name: str, llm_inference_client: MockLLMInferenceService, rag_service: MockRAGService,
                 ecommerce_api_client: ECommerceAPIClient, pii_masker: PIIMasker):
        self.name = name
        self.llm_inference_client = llm_inference_client
        self.rag_service = rag_service
        self.ecommerce_api_client = ecommerce_api_client
        self.pii_masker = pii_masker
        self.agent_reasoning_cache = {} # Cache2: for speeding up internal agent decisions

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