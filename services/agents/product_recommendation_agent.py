# services/agents/product_recommendation_agent.py
import logging
from typing import Any, Callable

from common.models import AgentTask, StructuredAgentResult, StructuredProductRecommendation, LLMAgentReasonRequest
from services.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SEARCH_ITEMS_TOOL = "ECommerceAPI.searchItems"
GET_POPULAR_CATEGORY_TOOL = "ECommerceAPI.getPopularCategory"

class ProductRecommendationAgent(BaseAgent):
    """
    Specialized AI Agent for generating product recommendations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("ProductRecommendationAgent", *args, **kwargs)

    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        """Recommend a product, either via a direct catalog tool call or,
        failing that, the RAG-based fallback below.

        Note the RAG-path recommendation itself (product_id, name, price) is
        still a hardcoded placeholder, not looked up from
        `clients/ecommerce_api_client.py`'s items table - only the RAG-matched
        description snippet and the customer-history framing (favorite
        category, last purchase) are real. The searchItems tool path below
        doesn't have this problem - it returns a real item_id/name/price.
        """
        logger.info("Received task: task_id=%s intent=%s", task.task_id, task.intent)

        user_id = task.user_id
        # 1. Fetch customer history (Internal Tool: E-commerce Microservice API)
        customer_history = self.ecommerce_api_client.get_customer_history(user_id)
        logger.debug("customer_history=%s", customer_history)

        # Tool registry, same convention as OrderTrackingAgent's - available_tools
        # and dispatch both derive from this dict. Unlike that agent's tools,
        # these take LLM-extracted params (reason_response.tool_params) since
        # there's no order_id/user_id to resolve upfront - the LLM has to read
        # what to search for out of the query itself.
        tools: dict[str, Callable[[dict], Any]] = {
            # keyword/query/search_term: the LLM doesn't reliably use the exact
            # param name asked for in the prompt (observed "query" as often as
            # "keyword") - accept the common synonyms rather than silently
            # dropping the term on a naming mismatch.
            SEARCH_ITEMS_TOOL: lambda params: self.ecommerce_api_client.search_items(
                category=params.get("category"),
                keyword=params.get("keyword") or params.get("query") or params.get("search_term"),
                limit=params.get("limit", 5),
            ),
            GET_POPULAR_CATEGORY_TOOL: lambda params: self.ecommerce_api_client.get_popular_category(
                limit=params.get("limit", 1)
            ),
        }

        reason_response = self.llm_inference_client.call_agent_reason(
            LLMAgentReasonRequest(
                session_id=task.session_id,
                agent_name=self.name,
                task_description=(
                    f"Handle this product-related request for customer {user_id} "
                    f"(favorite category: {customer_history.get('favorite_category')}, "
                    f"last purchase: {customer_history.get('last_purchase')}): \"{task.original_query}\""
                ),
                current_state={"user_id": user_id, **customer_history},
                available_tools=list(tools),
            )
        )
        logger.info(
            "Reasoning result: action=%s tool_name=%s thought=%s",
            reason_response.action, reason_response.tool_name, reason_response.thought,
        )

        if reason_response.action == "call_api" and reason_response.tool_name in tools:
            tool_result = tools[reason_response.tool_name](reason_response.tool_params or {})
            logger.info("EcommerceClient returned: %s", tool_result)

            if reason_response.tool_name == SEARCH_ITEMS_TOOL:
                if tool_result:
                    item = tool_result[0]
                    return StructuredAgentResult(
                        task_id=task.task_id,
                        agent_name=self.name,
                        status="success",
                        result_data=StructuredProductRecommendation(
                            product_id=item["item_id"],
                            name=item["name"],
                            description_snippet=(item.get("description") or "")[:150],
                            price=item["price"],
                            reason=f"Matched your search in our {item.get('category') or 'catalog'}.",
                        ),
                    )
                return StructuredAgentResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="failure",
                    result_data={"message": "No items matched that search. Try a different keyword or category."},
                )

            if reason_response.tool_name == GET_POPULAR_CATEGORY_TOOL:
                if tool_result:
                    top = tool_result[0]
                    return StructuredAgentResult(
                        task_id=task.task_id,
                        agent_name=self.name,
                        status="success",
                        result_data={
                            "category": top["category"],
                            "units_sold": top["units_sold"],
                            "message": f"Our most popular category right now is {top['category']}.",
                        },
                    )
                return StructuredAgentResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="failure",
                    result_data={"message": "Not enough order history yet to determine a popular category."},
                )

        # Fallback: original RAG-based flow (unchanged) for queries the LLM
        # didn't map to either tool above (e.g. an open-ended "recommend me
        # something" with no specific search term or category-popularity ask).
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