# services/agents/order_tracking_agent.py
import re
from common.models import AgentTask, StructuredAgentResult, StructuredOrderSummary, LLMAgentReasonRequest, LLMAgentInterpretRequest
from services.agents.base_agent import BaseAgent

ORDER_ID_PATTERN = re.compile(r"\b(\d{4,})\b")

class OrderTrackingAgent(BaseAgent):
    """
    Specialized AI Agent for handling order tracking inquiries.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("OrderTrackingAgent", *args, **kwargs)

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        match = ORDER_ID_PATTERN.search(text or "")
        return match.group(1) if match else None

    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        print(f"\n[{self.name}] Received task: {task.task_id} for intent '{task.intent}'")

        customer_id = task.customer_id # Assume Orchestrator has retrieved/tokenized this
        # The router never actually populates 'order_id' into task.params (it only
        # passes {"query": <text>}), so pull it out of the customer's own text instead
        # of silently defaulting to '12345' for every order.
        order_id = task.params.get('order_id') or self._extract_order_id(task.original_query) or '12345'

        # 1. Use LLMInf_AgentReason for structured workflow planning/tool selection
        # (This determines if we need to call an API, RAG, or return directly)
        reason_response = self.llm_inference_client.call_agent_reason(
            LLMAgentReasonRequest(
                session_id=task.session_id,
                agent_name=self.name,
                task_description=f"Find order details for order {order_id} for customer {customer_id}",
                current_state={'order_id': order_id, 'customer_id': customer_id},
                available_tools=['ECommerceAPI.getOrderDetails'] # Inform the LLM of available tools
            )
        )

        raw_order_details = {}
        if reason_response.action == 'call_api' and reason_response.tool_name == 'ECommerceAPI.getOrderDetails':
            # 2. Execute Internal Tool: E-commerce Microservice API call
            raw_order_details = self.ecommerce_api_client.get_order_details(customer_id, order_id)
        else:
            print(f"  [{self.name}] LLMInf_AgentReason decided not to call API: {reason_response.thought}")
        
        if not raw_order_details or "error" in raw_order_details:
            # Fallback if API call fails
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="failure",
                result_data={"message": f"Could not find details for order {order_id}. Please check the ID or try again."},
            )
        
        # 3. Use LLMInf_AgentInterpret for structured diagnosis/data analysis
        # (This takes raw API data and extracts deeper, structured meaning)
        interpret_response = self.llm_inference_client.call_agent_interpret(
            LLMAgentInterpretRequest(
                session_id=task.session_id,
                agent_name=self.name,
                raw_data=raw_order_details,
                interpretation_goal="diagnose order issue"
            )
        )
        
        # 4. Compile all findings into a final structured result
        order_summary = StructuredOrderSummary(
            order_id=raw_order_details.get('order_id'),
            status=raw_order_details.get('status'),
            items=raw_order_details.get('items', []),
            estimated_delivery=raw_order_details.get('estimated_delivery'),
            issue_analysis=interpret_response.structured_interpretation.get('issue_type') if isinstance(interpret_response.structured_interpretation, dict) else None
        )
        
        return StructuredAgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status="success",
            result_data=order_summary,
        )

# This agent would not typically be run directly but instantiated by the Orchestrator
if __name__ == "__main__":
    print("OrderTrackingAgent is a specialized agent.")
    # To test, you would need to mock all its dependencies.
