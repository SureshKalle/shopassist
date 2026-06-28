# services/agents/general_purpose_agent.py
from common.models import AgentTask, StructuredAgentResult
from services.agents.base_agent import BaseAgent

class GeneralPurposeAgent(BaseAgent):
    """
    Specialized AI Agent for handling general queries not covered by other specific agents.
    Primarily uses RAG for knowledge retrieval.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("GeneralPurposeAgent", *args, **kwargs)

    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        print(f"\n[{self.name}] Received task: {task.task_id} for intent '{task.intent}'")
        
        # 1. Try a RAG lookup for general knowledge
        rag_results = self.rag_service.query_knowledge_base(
            self.llm_inference_client.call_embeddings(task.original_query),
            task.original_query
        )
        
        if rag_results:
            # Return the content of the most relevant document
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="success",
                result_data={"answer_snippet": rag_results[0].content},
            )
        else:
            # If RAG finds nothing, provide a generic helpful response
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="success", # Still a success, just with a generic response
                result_data={"answer_snippet": f"I can help with general questions like: '{task.original_query}'. Please rephrase or ask something else."},
            )

# This agent would not typically be run directly
if __name__ == "__main__":
    print("GeneralPurposeAgent handles broad queries.")