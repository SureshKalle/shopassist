# services/agents/escalation_agent.py
import logging
from typing import Dict, Any # Added for explicit type hinting for params_dict

from pydantic import BaseModel # Added to check if task.params is a BaseModel

from common.models import AgentTask, StructuredAgentResult
from services.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class EscalationAgent(BaseAgent):
    """
    Specialized AI Agent designated as a fallback for unresolvable issues or critical failures.
    It prepares a structured summary for human agent handover.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("EscalationAgent", *args, **kwargs)

    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        # Robustly handle task.params, ensuring it's a dict for .get() access
        # If task.params is a Pydantic BaseModel, convert it to a dict using .model_dump()
        # Otherwise, assume it's already a dict or None
        params_dict: Dict[str, Any] = task.params.model_dump() if isinstance(task.params, BaseModel) else task.params if task.params is not None else {}

        escalation_reason = params_dict.get("reason", "Issue could not be resolved by automated agents.")
        logger.warning("Escalating: task_id=%s reason=%s", task.task_id, escalation_reason)
        
        # In a real system, this agent would also:
        # - Create a ticket in a CRM/Helpdesk system via E-commerce Microservices API.
        # - Notify a human agent or team.
        # - Potentially summarize the conversation history for the human agent.

        return StructuredAgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status="escalation",
            result_data={
                "escalation_reason": escalation_reason,
                "original_query": task.original_query,
                "conversation_summary": task.conversation_context[-3:] # Last 3 turns for context
            },
        )

# This agent would not typically be run directly
if __name__ == "__main__":
    logger.info("EscalationAgent handles unresolved queries.") # Changed print to logger.info for consistency
