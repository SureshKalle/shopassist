# services/data_pipeline.py
"""
Offline/batch ingestion pipeline that seeds services/rag.py's in-memory store
and would, in a real deployment, also feed LLM fine-tuning data. Not part of
the live request path - api/dependencies.py's warm_up_services() and
main_simulation.py both call this once at startup with a small hardcoded
sample dataset, not on every request.
"""
import logging
from typing import List
from common.models import (
    RawCustomerConversation, CleanedCustomerConversation,
    RawProductRecord, CleanedProductRecord,
    ChunkedDocument
)
from services.pii_masker import PIIMasker
from services.llm_inference import LLMInferenceService
from services.classifier_client import ClassifierClient
from services.rag import RAGPipeline
logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """
    Handles collecting, cleaning, anonymizing PII, chunking, and embedding data
    for RAG and LLM fine-tuning. This is typically a batch or streaming system,
    not a real-time microservice.
    """
    def __init__(self, pii_masker: PIIMasker, llm_inference_client: LLMInferenceService, rag_service: RAGPipeline,
                 classifier_client: ClassifierClient = None):
        self.pii_masker = pii_masker
        self.llm_inference_client = llm_inference_client
        self.rag_service = rag_service
        self.classifier_client = classifier_client or ClassifierClient()

    def ingest_customer_conversations(self, raw_conversations: List[RawCustomerConversation]) -> List[CleanedCustomerConversation]:
        logger.info("Ingesting %d customer conversation(s)", len(raw_conversations))
        cleaned_conversations = []
        for raw_conv in raw_conversations:
            logger.debug("Processing raw conversation ID: %s", raw_conv.id)
            
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
            """
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
            """
        logger.info("Customer conversation ingestion complete: %d record(s)", len(cleaned_conversations))
        return cleaned_conversations

    def ingest_product_catalog(self):
        logger.info("Ingesting product catalog into RAG store")
        status=self.rag_service.build_rag_pipeline()
        logger.info("RAG Pipeline Build Completed Successfully!")
        logger.info(f"Statistics: {status}")
        return status

                       
    def generate_synthetic_queries(self, base_queries: List[str]) -> List[str]:
        logger.info("Generating synthetic queries from %d base quer(y/ies)", len(base_queries))
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
        logger.info("Synthetic query generation complete: %d generated", len(synthetic_queries))
        return synthetic_queries

# Example of how this pipeline might be run (e.g., a scheduled Airflow job):
if __name__ == "__main__":
    pii_masker_inst = PIIMasker()
    llm_inf_inst = LLMInferenceService()
    rag_service_inst = MockRAGService(llm_inf_inst)
    
    pipeline = DataIngestionPipeline(pii_masker_inst, llm_inf_inst, rag_service_inst)

    raw_convs = [
        RawCustomerConversation(id="test_conv_001", text="Hello, my name is Jane Smith. I need help with order 54321.", metadata={"user_id": "alum_abc"}),
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
