# services/agents/product_recommendation_agent.py
import logging
from typing import Any, Callable

from common.models import AgentTask, StructuredAgentResult, StructuredProductRecommendation, LLMAgentReasonRequest
from services.agents.base_agent import BaseAgent
from services.llm_inference import EmbeddingUnavailableError
# --- Langfuse Integration Start ---
from langfuse import observe
# --- Langfuse Integration End ---

logger = logging.getLogger(__name__)

SEARCH_ITEMS_TOOL = "ECommerceAPI.searchItems"
GET_POPULAR_CATEGORY_TOOL = "ECommerceAPI.getPopularCategory"
ADD_REVIEW_TOOL = "ECommerceAPI.addReview"
REMOVE_REVIEW_TOOL = "ECommerceAPI.removeReview"

class ProductRecommendationAgent(BaseAgent):
    """
    Specialized AI Agent for generating product recommendations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__("ProductRecommendationAgent", *args, **kwargs)

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="product_recommendation_agent_process_task")
    # --- Langfuse Integration End ---
    def process_task(self, task: AgentTask) -> StructuredAgentResult:
        """Recommend a product, either via a direct catalog tool call or,
        failing that, the RAG-based fallback below.

        Both paths return a real item_id/name/price from
        `clients/ecommerce_api_client.py`'s items table - the RAG path uses
        the product_id ingest_product_catalog() (services/data_pipeline.py)
        stamped into the matched chunk's metadata to look the item up live,
        rather than trusting a value cached at ingestion time. Only the
        description snippet and the customer-history framing (favorite
        category, last purchase) come from RAG/history directly.
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
            # Same LLM param-naming looseness as searchItems above - accept the
            # common synonyms rather than dropping the review on a mismatch.
            # user_id is always the requesting customer, never LLM-supplied -
            # add_review()'s own verified-purchase guard is what actually
            # decides whether this succeeds, not anything extracted here.
            ADD_REVIEW_TOOL: lambda params: self.ecommerce_api_client.add_review(
                item_id=params.get("item_id") or params.get("product_id"),
                user_id=user_id,
                review_title=params.get("review_title") or params.get("title") or "",
                review_content=params.get("review_content") or params.get("content") or params.get("review") or "",
            ),
            REMOVE_REVIEW_TOOL: lambda params: self.ecommerce_api_client.remove_review(
                review_id=params.get("review_id"),
                user_id=user_id,
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

            if reason_response.tool_name == ADD_REVIEW_TOOL:
                if tool_result.get("error"):
                    return StructuredAgentResult(
                        task_id=task.task_id,
                        agent_name=self.name,
                        status="failure",
                        result_data={"message": tool_result["error"]},
                    )
                return StructuredAgentResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="success",
                    result_data={"message": "Thanks for your review!", **tool_result},
                )

            if reason_response.tool_name == REMOVE_REVIEW_TOOL:
                if tool_result.get("error"):
                    return StructuredAgentResult(
                        task_id=task.task_id,
                        agent_name=self.name,
                        status="failure",
                        result_data={"message": tool_result["error"]},
                    )
                return StructuredAgentResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="success",
                    result_data={"message": "Your review has been deleted.", **tool_result},
                )

        # Fallback: original RAG-based flow (unchanged) for queries the LLM
        # didn't map to either tool above (e.g. an open-ended "recommend me
        # something" with no specific search term or category-popularity ask).
        # 2. Use RAG to find relevant products based on query and history (Internal Tool: RAG Service)
        rag_query_text = f"{task.original_query} based on customer's favorite category '{customer_history.get('favorite_category')}' and last purchase '{customer_history.get('last_purchase')}'"
        # If the embedding backend is down, treat it the same as "RAG found
        # nothing" (below) rather than letting the exception crash this
        # customer's whole chat turn - see EmbeddingUnavailableError's
        # docstring (services/llm_inference.py) for why call_embeddings()
        # raises instead of returning a meaningless zero-vector here.
        try:
            rag_results = self.rag_service.query_knowledge_base(
                self.llm_inference_client.call_embeddings(rag_query_text),
                rag_query_text,
                top_k=3 # Request multiple relevant documents
            )
        except EmbeddingUnavailableError:
            logger.error("process_task: embedding backend unavailable for RAG fallback query - task_id=%s", task.task_id, exc_info=True)
            rag_results = []
        
        recommended_product = None
        if rag_results:
            # In a real system, LLMInf_AgentInterpret might help select the best product and reason
            product_info_doc = next((doc for doc in rag_results if doc.source_type == 'product_catalog'), None)
            if product_info_doc:
                # ingest_product_catalog() (services/data_pipeline.py) stamps every
                # product_catalog chunk with its real product_id in metadata at
                # ingestion time - reuse it to look up current name/price live
                # rather than caching them here, so a price change since ingestion
                # doesn't get echoed back stale.
                real_product_id = product_info_doc.metadata.get("product_id")
                item = self.ecommerce_api_client.get_item(real_product_id) if real_product_id else None
                if item:
                    recommended_product = StructuredProductRecommendation(
                        product_id=item["item_id"],
                        name=item["name"],
                        description_snippet=product_info_doc.content[:150], # Use snippet from RAG
                        price=item["price"],
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