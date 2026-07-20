# services/data_pipeline.py
"""
Offline/batch ingestion pipeline that seeds services/rag.py's in-memory store
and would, in a real deployment, also feed LLM fine-tuning data. Not part of
the live request path - api/dependencies.py's warm_up_services() and
main_simulation.py both call this once at startup with a small hardcoded
sample dataset, not on every request.
"""
import logging
import os # NEW: Added for file system operations
import uuid # NEW: Added for generating unique IDs for chunks
from typing import List, Dict, Any # Added Any for TextLoader metadata
# NEW IMPORTS for PDF processing and text splitting
from pypdf import PdfReader # For reading PDF content
from langchain_text_splitters import RecursiveCharacterTextSplitter # For robust text chunking
# END NEW IMPORTS

from common.models import (
    RawCustomerConversation, CleanedCustomerConversation,
    RawProductRecord, CleanedProductRecord,
    ChunkedDocument
)
from services.pii_masker import PIIMasker
from services.llm_inference import LLMInferenceService
from services.classifier_client import ClassifierClient
# --- MODIFIED: Import RAGService, not MockRAGService ---
from services.rag import RAGService
# --- END MODIFIED ---

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """
    Handles collecting, cleaning, anonymizing PII, chunking, and embedding data
    for RAG and LLM fine-tuning. This is typically a batch or streaming system,
    not a real-time microservice.
    """
    # --- MODIFIED: Use RAGService type hint ---
    def __init__(self, pii_masker: PIIMasker, llm_inference_client: LLMInferenceService, rag_service: RAGService,
                 classifier_client: ClassifierClient = None):
        self.pii_masker = pii_masker
        self.llm_inference_client = llm_inference_client
        self.rag_service = rag_service
        self.classifier_client = classifier_client or ClassifierClient()
        # NEW: Initialize text splitter for general documents (like PDFs)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,   # Example chunk size
            chunk_overlap=200, # Example overlap
            length_function=len,
        )
        # END NEW

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
        logger.info("Customer conversation ingestion complete: %d record(s)", len(cleaned_conversations))
        return cleaned_conversations

    def ingest_product_catalog(self, raw_products: List[RawProductRecord]) -> List[CleanedProductRecord]:
        logger.info("Ingesting %d product record(s)", len(raw_products))
        cleaned_products = []
        for raw_prod in raw_products:
            logger.debug("Processing raw product ID: %s", raw_prod.product_id)

            # --- 1. Cleaning & Organization ---
            clean_description = raw_prod.raw_description.strip()
            # Handle missing values: e.g., default values, imputation
            # De-duplicate product listings: based on unique identifiers or content hashing
            # Normalize variations: e.g., "CPU: i7" vs "Processor: Intel Core i7" -> "cpu": "intel_core_i7"
            # Reconcile inconsistent price formats: already done via float conversion
            try:
                normalized_price = float(raw_prod.price.replace('₹', '').replace('$', '').replace(',', ''))
            except ValueError:
                normalized_price = 0.0 # Assign default or flag as error

            # Extract structured entities from free-text (e.g., features from description)
            structured_specs = {k.lower().replace(' ', '_'): v for k, v in raw_prod.specs.items()}

            # Process reviews: de-duplicate, standardize, sentiment analysis
            sentiment_analyzed_reviews = []
            for review_text in set(raw_prod.reviews): # Use set to de-duplicate
                # Real classifier call (services/classifier_client.py) - fails
                # soft to label="unknown" if that service isn't running, so
                # ingestion never breaks on it being unavailable.
                sentiment = self.classifier_client.classify_sentiment(review_text)

                # --- 2. PII Masking for Reviews ---
                masked_review_data = self.pii_masker.mask_text(review_text)
                sentiment_analyzed_reviews.append({"text": masked_review_data.masked_text, "sentiment": sentiment.label, "sentiment_score": sentiment.score, "original_hash": masked_review_data.original_text_hash})

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
        logger.info("Product catalog ingestion complete: %d record(s)", len(cleaned_products))
        return cleaned_products

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

    # --- NEW METHOD: ingest_pdf_documents ---
    def ingest_pdf_documents(self, docs_folder: str, source_type: str = "internal_document") -> int:
        """
        Reads all PDF documents from the specified folder, chunks them,
        applies PII masking, generates embeddings, and ingests them into the RAG service.
        """
        if not os.path.isdir(docs_folder):
            logger.error("Provided docs_folder '%s' is not a valid directory.", docs_folder)
            return 0

        logger.info("Starting PDF document ingestion from folder: '%s'", docs_folder)
        processed_chunks_count = 0

        for file_name in os.listdir(docs_folder):
            if file_name.lower().endswith(".pdf"):
                file_path = os.path.join(docs_folder, file_name)
                logger.debug("Processing PDF file: %s", file_path)

                try:
                    reader = PdfReader(file_path)
                    raw_text = ""
                    for page in reader.pages:
                        raw_text += page.extract_text() or "" # Extract text, handle None case

                    if not raw_text.strip():
                        logger.warning("PDF file '%s' had no extractable text. Skipping.", file_name)
                        continue

                    # --- PII Masking of Raw PDF Text ---
                    # Mask entire document first, as chunks might split PII
                    masked_pdf_data = self.pii_masker.mask_text(raw_text, session_id=f"pdf_ingest_{file_name}", user_id="system_ingest")
                    masked_text = masked_pdf_data.masked_text

                    # --- Chunking the Masked Text ---
                    chunks = self.text_splitter.split_text(masked_text)
                    
                    if not chunks:
                        logger.warning("No chunks generated from PDF file '%s' after masking. Skipping.", file_name)
                        continue

                    for i, chunk_content in enumerate(chunks):
                        if not chunk_content.strip(): # Skip empty chunks
                            continue

                        chunk_id = f"{source_type}_{file_name.replace('.', '_')}_chunk_{i}"
                        
                        # --- Embedding for RAG ---
                        embedding = self.llm_inference_client.call_embeddings(chunk_content)

                        chunk_document = ChunkedDocument(
                            doc_id=chunk_id,
                            content=chunk_content,
                            embedding=embedding,
                            source_type=source_type,
                            metadata={"file_name": file_name, "chunk_idx": i, "original_text_hash": masked_pdf_data.original_text_hash}
                        )
                        self.rag_service.ingest_document(chunk_document)
                        processed_chunks_count += 1
                    logger.info("Successfully processed '%s', generated %d chunks.", file_name, len(chunks))

                except Exception as e:
                    logger.error("Error processing PDF file '%s': %s", file_name, e, exc_info=True)
        
        logger.info("PDF document ingestion complete. Total %d chunks processed and ingested.", processed_chunks_count)
        return processed_chunks_count
    # --- END NEW METHOD ---


# Example of how this pipeline might be run (e.g., a scheduled Airflow job):
if __name__ == "__main__":
    import time
    # --- Set up a dummy docs folder and PDF for demonstration ---
    DOCS_FOLDER = "docs"
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    dummy_pdf_path = os.path.join(DOCS_FOLDER, "example_policy.pdf")

    # Create a simple dummy PDF file for testing
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    try:
        c = canvas.Canvas(dummy_pdf_path, pagesize=letter)
        c.drawString(100, 750, "E-commerce Return Policy")
        c.drawString(100, 700, "1. Returns are accepted within 30 days of purchase.")
        c.drawString(100, 680, "2. Items must be in original condition with packaging.")
        c.drawString(100, 660, "3. Refunds are processed within 5-7 business days.")
        c.drawString(100, 640, "4. Contact support@example.com for assistance. My name is John Doe.")
        c.save()
        print(f"Created dummy PDF: {dummy_pdf_path}")
    except Exception as e:
        print(f"Could not create dummy PDF (ReportLab might not be installed or error): {e}. Skipping PDF ingestion demo.")
        dummy_pdf_path = None # Disable PDF demo if creation failed
    # --- End dummy PDF setup ---

    pii_masker_inst = PIIMasker()
    llm_inf_inst = LLMInferenceService()
    # --- MODIFIED: Use RAGService, not MockRAGService, and specify file paths ---
    rag_data_file = "demo_rag_knowledge_base.jsonl"
    faiss_index_file = "demo_faiss_index.bin"

    # Clean up previous demo files for a fresh run
    if os.path.exists(rag_data_file):
        os.remove(rag_data_file)
        print(f"Removed existing {rag_data_file} for a clean demo run.")
    if os.path.exists(faiss_index_file):
        os.remove(faiss_index_file)
        print(f"Removed existing {faiss_index_file} for a clean demo run.")

    rag_service_inst = RAGService(llm_inf_inst, db_file_path=rag_data_file, faiss_index_path=faiss_index_file)
    print(f"\nInitialized RAGService for DataPipeline. JSONL file: {rag_service_inst.db_file_path}, FAISS index: {rag_service_inst.faiss_index_path}")
    # --- END MODIFIED ---

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
    # The 'masked_review_samples' metadata was a mock-specific print, removed for consistency
    # print(f"Sample Masked Review: {cleaned_prods[0].metadata['masked_review_samples'][0]}")
    print(f"RAG doc_store has {len(rag_service_inst.doc_store)} documents after pipeline run.")

    # --- NEW: Call the new PDF ingestion method ---
    if dummy_pdf_path and os.path.exists(dummy_pdf_path): # Only run if PDF was successfully created
        print("\n--- Ingesting PDF Documents ---")
        num_pdf_chunks = pipeline.ingest_pdf_documents(docs_folder=DOCS_FOLDER, source_type="customer_policy")
        print(f"Ingested {num_pdf_chunks} chunks from PDF documents.")
        print(f"RAG doc_store has {len(rag_service_inst.doc_store)} documents after PDF ingestion.")
    # --- END NEW ---

    print("\n--- Testing RAG Retrieval with New Data ---")
    # Simulate a query that would hit the new PDF content
    query_text_pdf = "How do I return something and what is the email for support?"
    query_embedding_pdf = llm_inf_inst.call_embeddings(query_text_pdf)
    
    rag_results_pdf = rag_service_inst.query_knowledge_base(query_embedding_pdf, query_text_pdf, top_k=2)
    print(f"\nSemantic Query Results for '{query_text_pdf}':")
    if rag_results_pdf:
        for i, res in enumerate(rag_results_pdf):
            print(f"- {i+1}. Doc ID: {res.doc_id}, Content: '{res.content[:70]}...' (Source: {res.source_type})")
            if 'email' in res.content.lower():
                print(f"    (Email found in chunk: {res.content})")
    else:
        print("No results found for PDF query.")

    print("\n--- Simulation Complete ---")

    # --- Clean up dummy PDF and folder ---
    if dummy_pdf_path and os.path.exists(dummy_pdf_path):
        os.remove(dummy_pdf_path)
        print(f"Cleaned up dummy PDF: {dummy_pdf_path}")
    if os.path.exists(DOCS_FOLDER) and not os.listdir(DOCS_FOLDER): # Only remove if empty
        os.rmdir(DOCS_FOLDER)
        print(f"Removed empty docs folder: {DOCS_FOLDER}")
    # --- End cleanup ---