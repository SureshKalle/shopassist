# services/agents/product_recommendation_agent.py
from common.models import AgentTask, StructuredAgentResult, StructuredProductRecommendation
from services.agents.base_agent import BaseAgent

class ProductRecommendationAgent(BaseAgent):
    """
    Specialized AI Agent for generating product recommendations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("ProductRecommendationAgent", *args, **kwargs)

    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        print(f"\n[{self.name}] Received task: {task.task_id} for intent '{task.intent}'")
        
        customer_id = task.customer_id
        # 1. Fetch customer history (Internal Tool: E-commerce Microservice API)
        customer_history = self.ecommerce_api_client.get_customer_history(customer_id)
        
        # 2. Use RAG to find relevant products based on query and history (Internal Tool: RAG Service)
        rag_query_text = f"{task.original_query} based on customer's favorite category '{customer_history.get('favorite_category')}' and last purchase '{customer_history.get('last_purchase')}'"
        rag_results = self.rag_service.query_knowledge_base(
            self.llm_inference_client.call_embeddings(rag_query_text),
            rag_query_text,
            top_k=3 # Request multiple relevant documents
        )
        
        recommended_product = None
        if rag_results:
            # Simulate picking a product from RAG results and crafting a structured recommendation
            # In a real system, LLMInf_AgentInterpret might help select the best product and reason
            product_info_doc = next((doc for doc in rag_results if doc.source_type == 'product_catalog'), None)
            if product_info_doc:
                # Mock a structured recommendation
                recommended_product = StructuredProductRecommendation(
                    product_id="PROD_XYZ", # This would be extracted from product_info_doc
                    name="Super Widget Pro",
                    description_snippet=product_info_doc.content[:50] + "...", # Use snippet from RAG
                    price=299.99, # This would be looked up from a product DB based on product_id
                    reason=f"it aligns with your interest in {customer_history.get('favorite_category')} and similar to your last purchase '{customer_history.get('last_purchase', 'no previous purchase')}' as suggested by our knowledge base."
                )
        
        if recommended_product:
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="success",
                result_data=recommended_product,
            )
        else:
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="failure",
                result_data={"message": "Could not find a suitable recommendation at this time. Please provide more details or try again later."},
            )

# This agent would not typically be run directly
if __name__ == "__main__":
    print("ProductRecommendationAgent is a specialized agent.")