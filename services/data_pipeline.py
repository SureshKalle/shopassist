# services/data_pipeline.py
from typing import List
from common.models import (
    RawCustomerConversation, CleanedCustomerConversation,
    RawProductRecord, CleanedProductRecord,
    ChunkedDocument
)
from services.pii_masker import PIIMasker
from services.llm_inference import LLMInferenceService
from services.rag import MockRAGService

class DataIngestionPipeline:
    """
    Handles collecting, cleaning, anonymizing PII, chunking, and embedding data
    for RAG and LLM fine-tuning. This is typically a batch or streaming system,
    not a real-time microservice.
    """
    def __init__(self, pii_masker: PIIMasker, llm_inference_client: LLMInferenceService, rag_service: MockRAGService):
        self.pii_masker = pii_masker
        self.llm_inference_client = llm_inference_client
        self.rag_service = rag_service

    def ingest_customer_conversations(self, raw_conversations: List[RawCustomerConversation]) -> List[CleanedCustomerConversation]:
        print("\n--- Data Pipeline: Ingesting Customer Conversations ---")
        cleaned_conversations = []
        for raw_conv in raw_conversations:
            print(f"  Processing raw conversation ID: {raw_conv.id}")
            
            # --- 1. Cleaning & Organization (e.g., lowercasing, removing noise) ---
            cleaned_text = raw_conv.text.lower().strip()
            # Advanced steps here:
            # - Typos/Grammar correction: using NLP libraries (e.g., pyspellchecker, textblob)
            # - Slang/informal language standardization: custom dictionaries or LLM-based normalization
            # - Conversation turn parsing: identify speaker, turns, dialogue acts for structured history
            # - Noise removal: remove HTML tags, URLs, irrelevant system messages
            
            # --- 2. PII Masking ---
            # Crucial step: PII is masked before any data reaches an LLM (even for embeddings/fine-tuning)
            masked_data = self.pii_masker.mask_text(cleaned_text, session_id=raw_conv.id, user_id=raw_conv.metadata.get('user_id'))
            masked_text = masked_data.masked_text
            
            # Store for potential LLM fine-tuning (e.g., in a data lake/warehouse)
            cleaned_conversations.append(CleanedCustomerConversation(
                id=raw_conv.id,
                cleaned_text=masked_text,
                tokens=masked_text.split(), # Simple tokenization; in production, use a tokenizer
                metadata={**raw_conv.metadata, "original_text_hash": masked_data.original_text_hash}
            ))

            # --- 3. Chunking and Embedding for RAG ---
            # Only chunk/embed if content is meaningful after masking
            if masked_text and len(masked_text) > 20: # Example length threshold
                chunks = [masked_text] # Simple: one chunk per conversation; real: recursive chunking
                for i, chunk_content in enumerate(chunks):
                    chunk = ChunkedDocument(
                        doc_id=f"conv_chunk_{raw_conv.id}_{i}",
                        content=chunk_content,
                        embedding=self.llm_inference_client.call_embeddings(chunk_content),
                        source_type="customer_support_conversation",
                        metadata={"conv_id": raw_conv.id, "chunk_idx": i}
                    )
                    self.rag_service.ingest_document(chunk)
        print("--- Customer Conversation Ingestion Complete ---")
        return cleaned_conversations

    def ingest_product_catalog(self, raw_products: List[RawProductRecord]) -> List[CleanedProductRecord]:
        print("\n--- Data Pipeline: Ingesting Product Catalog ---")
        cleaned_products = []
        for raw_prod in raw_products:
            print(f"  Processing raw product ID: {raw_prod.product_id}")
            
            # --- 1. Cleaning & Organization ---
            clean_description = raw_prod.raw_description.strip()
            # Handle missing values: e.g., default values, imputation
            # De-duplicate product listings: based on unique identifiers or content hashing
            # Normalize variations: e.g., "CPU: i7" vs "Processor: Intel Core i7" -> "cpu": "intel_core_i7"
            # Reconcile inconsistent price formats: already done via float conversion
            try:
                normalized_price = float(raw_prod.price.replace('$', '').replace(',', ''))
            except ValueError:
                normalized_price = 0.0 # Assign default or flag as error

            # Extract structured entities from free-text (e.g., features from description)
            structured_specs = {k.lower().replace(' ', '_'): v for k, v in raw_prod.specs.items()}
            
            # Process reviews: de-duplicate, standardize, sentiment analysis
            sentiment_analyzed_reviews = []
            for review_text in set(raw_prod.reviews): # Use set to de-duplicate
                # Mock sentiment: In reality, use a pre-trained sentiment model
                sentiment = "positive" if "great" in review_text.lower() or "awesome" in review_text.lower() else "negative" if "bad" in review_text.lower() else "neutral"
                
                # --- 2. PII Masking for Reviews ---
                masked_review_data = self.pii_masker.mask_text(review_text)
                sentiment_analyzed_reviews.append({"text": masked_review_data.masked_text, "sentiment": sentiment, "original_hash": masked_review_data.original_text_hash})
            
            cleaned_products.append(CleanedProductRecord(
                product_id=raw_prod.product_id,
                clean_description=clean_description,
                structured_specs=structured_specs,
                sentiment_analyzed_reviews=sentiment_analyzed_reviews,
                normalized_price=normalized_price,
                metadata={"original_price_str": raw_prod.price}
            ))

            # --- 3. Chunking and Embedding for RAG ---
            # Combine relevant info for retrieval: description, key specs, summary of positive/negative reviews
            reviews_summary = ". ".join([r['text'] for r in sentiment_analyzed_reviews])
            content_for_rag = f"Product: {clean_description}. Specifications: {raw_prod.specs}. Customer Reviews: {reviews_summary}"
            
            chunk = ChunkedDocument(
                doc_id=f"prod_chunk_{raw_prod.product_id}",
                content=content_for_rag,
                embedding=self.llm_inference_client.call_embeddings(content_for_rag),
                source_type="product_catalog",
                metadata={"product_id": raw_prod.product_id}
            )
            self.rag_service.ingest_document(chunk)
        print("--- Product Catalog Ingestion Complete ---")
        return cleaned_products

    def generate_synthetic_queries(self, base_queries: List[str]) -> List[str]:
        print("\n--- Data Pipeline: Generating Synthetic Queries ---")
        synthetic_queries = []
        # In a real system, GPT-5 or similar would generate variations (paraphrasing, adding details, varying tone)
        # This can be done by sending requests to a powerful LLM like LLMInf_Generative or an external API.
        for query in base_queries:
            # Mocking generation
            synthetic_queries.append(query + " - generated variant (formal)")
            synthetic_queries.append(query + " - generated variant (casual)")
        
        # --- Quality Filtering for Synthetic Data ---
        # After generation, filter out low-quality, nonsensical, or redundant synthetic queries.
        # This could involve heuristics, another small LLM for quality scoring, or diversity metrics.
        print("--- Synthetic Query Generation Complete ---")
        return synthetic_queries

# Example of how this pipeline might be run (e.g., a scheduled Airflow job):
if __name__ == "__main__":
    pii_masker_inst = PIIMasker()
    llm_inf_inst = LLMInferenceService()
    rag_service_inst = MockRAGService(llm_inf_inst)
    
    pipeline = DataIngestionPipeline(pii_masker_inst, llm_inf_inst, rag_service_inst)

    raw_convs = [
        RawCustomerConversation(id="test_conv_001", text="Hello, my name is Jane Smith. I need help with order 54321.", metadata={"user_id": "cust_abc"}),
    ]
    cleaned_convs = pipeline.ingest_customer_conversations(raw_convs)
    print(f"\nSample Cleaned Conv: {cleaned_convs[0].cleaned_text}")

    raw_prods = [
        RawProductRecord(product_id="TEST_PROD_1", raw_description="A fantastic test product.", specs={"Weight": "1kg"}, reviews=["Amazing!", "Totally worth it, Jane Smith!"], price="€99.99"),
    ]
    cleaned_prods = pipeline.ingest_product_catalog(raw_prods)
    print(f"\nSample Cleaned Product: {cleaned_prods[0].clean_description}")
    print(f"Sample Masked Review: {cleaned_prods[0].metadata['masked_review_samples'][0]}")
    print(f"RAG has {len(rag_service_inst.vector_db)} documents after pipeline run.")
