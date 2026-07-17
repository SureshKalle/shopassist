# services/agents/product_recommendation_agent.py
import logging

from common.models import AgentTask, StructuredAgentResult, StructuredProductRecommendation
from services.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ProductRecommendationAgent(BaseAgent):
    """
    Specialized AI Agent for generating product recommendations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("ProductRecommendationAgent", *args, **kwargs)

    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        """Recommend one product based on the customer's purchase history + RAG.

        Note the recommendation itself (product_id, name, price below) is
        currently a hardcoded placeholder, not looked up from
        `clients/ecommerce_api_client.py`'s items table - only the RAG-matched
        description snippet and the customer-history framing (favorite
        category, last purchase) are real. Wiring product_id/price/name to a
        real `EcommerceClient.get_item()`/`search_items()` call is a natural
        next step once RAG returns enough to identify a specific item_id.
        """
        logger.info("Received task: task_id=%s intent=%s", task.task_id, task.intent)

        user_id = task.user_id
        # 1. Fetch customer history (Internal Tool: E-commerce Microservice API)
        customer_history = self.ecommerce_api_client.get_customer_history(user_id)
        logger.debug("customer_history=%s", customer_history)

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
            logger.info("Recommendation found: product_id=%s", recommended_product.product_id)
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="success",
                result_data=recommended_product,
            )
        else:
            logger.warning("No recommendation found for task_id=%s", task.task_id)
            return StructuredAgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="failure",
                result_data={"message": "Could not find a suitable recommendation at this time. Please provide more details or try again later."},
            )

# This agent would not typically be run directly
if __name__ == "__main__":
    print("ProductRecommendationAgent is a specialized agent.")