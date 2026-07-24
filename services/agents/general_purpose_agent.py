# services/agents/general_purpose_agent.py
import logging

from common.models import AgentTask, StructuredAgentResult
from services.agents.base_agent import BaseAgent

# --- Langfuse Integration Start ---
from langfuse import observe
# --- Langfuse Integration End ---

logger = logging.getLogger(__name__)

class GeneralPurposeAgent(BaseAgent):
    """
    Specialized AI Agent for handling general queries not covered by other specific agents.
    Primarily uses RAG for knowledge retrieval.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("GeneralPurposeAgent", *args, **kwargs)

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="general_purpose_agent_process_task")
    # --- Langfuse Integration End ---
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        logger.info("Received task: task_id=%s intent=%s", task.task_id, task.intent)

        # 1. Try a RAG lookup for general knowledge. top_k=3 (not the
        # default 1, matching ProductRecommendationAgent's own reasoning
        # below) - policy PDFs chunk into ~2 pieces each (services/data_pipeline.py's
        # chunk_size=1000), so a single top-1 chunk risks missing the other
        # half of one policy, or dropping an entire topic on a combined
        # multi-topic question (e.g. "return policy AND shipping times").
        rag_results = self.rag_service.query_knowledge_base(
            self.llm_inference_client.call_embeddings(task.original_query),
            task.original_query,
            top_k=3,
        )

        if rag_results:
            logger.info("RAG match found for task_id=%s (%d chunk(s))", task.task_id, len(rag_results))
            # Join all matched chunks (not just the single closest one) so
            # call_generative() gets full context to synthesize from.
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="success",
                result_data={"answer_snippet": "\n\n".join(r.content for r in rag_results)},
            )
        else:
            logger.info("No RAG match for task_id=%s - returning generic response", task.task_id)
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