# services/rag.py
"""
FAISS-powered vector store for RAG. Documents are ingested via ingest_document()
(services/data_pipeline.py and api/dependencies.py's warm_up_services() both
do this at startup).

Retrieval now uses FAISS for vector similarity search against the document
embeddings, and the JSON Lines file (`rag_knowledge_base.jsonl`) serves as
the persistent store for the full ChunkedDocument objects (content + metadata)
which are mapped by their integer index in the FAISS vector index.
"""
import logging
import json
import os
import faiss       # NEW: Import faiss for vector indexing
import numpy as np # NEW: Import numpy for array manipulation (FAISS input/output)
from typing import List, Dict, Optional, Any
from common.models import ChunkedDocument
from services.llm_inference import LLMInferenceService
# --- Langfuse Integration Start ---
from langfuse import observe
# --- Langfuse Integration End ---

logger = logging.getLogger(__name__)

# --- MODIFIED DEFAULT_EMBEDDING_DIM ---
# Set default embedding dimension to match nomic-embed-text (3072)
DEFAULT_EMBEDDING_DIM = 3072
# --- END MODIFIED ---

class RAGService:
    """
    Implements a RAG Service using FAISS for vector retrieval and
    a JSON Lines file for persistent storage of ChunkedDocument data.
    """
    def __init__(self, llm_inference_client: LLMInferenceService,
                 db_file_path: str = "rag_knowledge_base.jsonl",
                 faiss_index_path: str = "faiss_index.bin"):
        self.llm_inference_client = llm_inference_client
        self.db_file_path = db_file_path
        self.faiss_index_path = faiss_index_path

        self.doc_store: List[ChunkedDocument] = [] # Stores ChunkedDocument objects, indexed by list position
        self.faiss_index: Optional[faiss.Index] = None # The FAISS vector index

        # --- MODIFIED: Setup FAISS index at initialization ---
        self._setup_faiss_index()
        # --- END MODIFIED ---

        # The rag_query_cache is still kept, could be adapted for embedding hash or removed if FAISS is fast enough
        self.rag_query_cache: Dict[str, List[ChunkedDocument]] = {}


    # --- NEW: Methods for FAISS setup and JSONL persistence ---
    def _load_from_file(self) -> List[ChunkedDocument]:
        """Loads ChunkedDocuments from the specified JSON Lines file."""
        loaded_docs: List[ChunkedDocument] = []
        if not self.db_file_path or not os.path.exists(self.db_file_path):
            logger.info("RAG DB file not found at %s. Starting with an empty doc store.", self.db_file_path)
            return loaded_docs

        logger.info("Loading ChunkedDocuments from %s...", self.db_file_path)
        with open(self.db_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    doc = ChunkedDocument.model_validate(data) # Pydantic validation
                    loaded_docs.append(doc)
                except json.JSONDecodeError as e:
                    logger.error("Error decoding JSON on line %d in RAG DB file (%s): %s - Line content: %s", line_num, self.db_file_path, e, line.strip())
                except Exception as e:
                    logger.error("Error validating ChunkedDocument on line %d from RAG DB file (%s): %s - Line content: %s", line_num, self.db_file_path, e, line.strip())
        logger.info("Loaded %d ChunkedDocuments from %s.", len(loaded_docs), self.db_file_path)
        return loaded_docs

    def _append_to_file(self, doc: ChunkedDocument):
        """Appends a single ChunkedDocument to the JSON Lines file."""
        if not self.db_file_path:
            logger.warning("No db_file_path set for RAGService, skipping file write.")
            return
        try:
            with open(self.db_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(doc.model_dump()) + '\n')
            logger.debug("Appended document '%s' to %s", doc.doc_id, self.db_file_path)
        except Exception as e:
            logger.error("Failed to append document '%s' to file %s: %s", doc.doc_id, self.db_file_path, e)

    def _setup_faiss_index(self):
        """
        Initializes the FAISS index by loading from disk or building from loaded documents.
        This method is called once at service startup.
        """
        self.doc_store = self._load_from_file() # Populate doc_store from JSONL file

        if os.path.exists(self.faiss_index_path):
            try:
                logger.info("Loading FAISS index from %s...", self.faiss_index_path)
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                logger.info("FAISS index loaded successfully with %d vectors.", self.faiss_index.ntotal)
                # Verify consistency: If doc_store and FAISS index have different sizes, something is off.
                # For this dev setup, we trust the JSONL is the source of truth for docs.
                if self.faiss_index.ntotal != len(self.doc_store):
                    logger.warning("FAISS index has %d vectors, but doc_store has %d documents. Index might be out of sync. Rebuilding for safety.", self.faiss_index.ntotal, len(self.doc_store))
                    self._build_faiss_index_from_doc_store() # Rebuild to ensure consistency
            except Exception as e:
                logger.error("Failed to load FAISS index from %s: %s. Attempting to rebuild.", self.faiss_index_path, e)
                self.faiss_index = None # Reset and try to rebuild
                self._build_faiss_index_from_doc_store() # Rebuild
        else:
            logger.info("FAISS index file not found at %s. Attempting to build new index.", self.faiss_index_path)
            self._build_faiss_index_from_doc_store()

    def _build_faiss_index_from_doc_store(self):
        """Builds a new FAISS index from the current in-memory doc_store."""
        if not self.doc_store:
            logger.warning("No documents in doc_store to build FAISS index. Initializing empty index.")
            self.faiss_index = faiss.IndexFlatL2(DEFAULT_EMBEDDING_DIM) # Initialize empty index
            return

        # Ensure all embeddings have the same dimension, otherwise FAISS will fail
        embedding_dimension = len(self.doc_store[0].embedding) if self.doc_store else DEFAULT_EMBEDDING_DIM
        if not all(len(doc.embedding) == embedding_dimension for doc in self.doc_store):
            logger.error("Embeddings have inconsistent dimensions. Cannot build FAISS index. Initializing empty index.")
            # Fallback to an empty index or raise an error
            self.faiss_index = faiss.IndexFlatL2(DEFAULT_EMBEDDING_DIM)
            return
        
        # --- NEW: Check if index is already correctly initialized with current dimension ---
        if self.faiss_index and self.faiss_index.d == embedding_dimension and self.faiss_index.ntotal == len(self.doc_store):
             logger.info("FAISS index already consistent with doc_store, no rebuild needed.")
             return
        # --- END NEW ---

        logger.info("Building new FAISS index from %d documents with dimension %d...", len(self.doc_store), embedding_dimension)
        self.faiss_index = faiss.IndexFlatL2(embedding_dimension) # Using L2 (Euclidean) distance
        
        # Prepare embeddings for FAISS (needs numpy array of float32)
        embeddings_matrix = np.array([doc.embedding for doc in self.doc_store]).astype('float32')
        self.faiss_index.add(embeddings_matrix)
        
        faiss.write_index(self.faiss_index, self.faiss_index_path)
        logger.info("FAISS index built and saved to %s with %d vectors.", self.faiss_index_path, self.faiss_index.ntotal)
    # --- END NEW methods ---


    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="rag_service_query_knowledge_base")
    # --- Langfuse Integration End ---
    # --- MODIFIED: Use FAISS for retrieval ---
    def query_knowledge_base(self, query_embedding: List[float], query_text: str, top_k: int = 1) -> List[ChunkedDocument]:
        """
        Performs vector similarity search using FAISS to retrieve top_k documents.
        `query_text` is still included for logging/context but the search is
        based on `query_embedding`.
        """
        logger.info("query_knowledge_base: query=%r top_k=%d (using FAISS)", query_text, top_k)

        if not self.faiss_index or self.faiss_index.ntotal == 0:
            logger.warning("FAISS index not initialized or empty. Cannot perform semantic search. Returning empty results.")
            return []

        # Convert query embedding to numpy array (FAISS expects float32)
        query_vector = np.array([query_embedding]).astype('float32')

        # --- FIX: Check query_vector dimension against FAISS index dimension ---
        if query_vector.shape[1] != self.faiss_index.d:
            logger.error("Query embedding dimension mismatch. FAISS index expects %d, got %d. Returning empty results.", self.faiss_index.d, query_vector.shape[1])
            return []
        # --- END FIX ---

        # Perform similarity search (distances, indices)
        # D: distances of the k nearest neighbors
        # I: labels (indices) of the k nearest neighbors
        distances, indices = self.faiss_index.search(query_vector, top_k)

        results: List[ChunkedDocument] = []
        for i_idx in indices[0]: # indices[0] contains the indices of the top_k most similar documents
            if i_idx == -1: # FAISS returns -1 for empty slots if top_k > ntotal
                continue
            if 0 <= i_idx < len(self.doc_store): # Defensive check against out-of-bounds indices
                results.append(self.doc_store[i_idx])
            else:
                logger.warning("FAISS returned an out-of-bounds index: %d. Index might be stale or doc_store mismatch.", i_idx)

        logger.info("query_knowledge_base: %d result(s) for %r (FAISS search completed)", len(results), query_text)
        # The results from FAISS are already sorted by similarity, so no extra sort needed.
        return results
    # --- END MODIFIED ---

    # --- Langfuse Integration Start: @observe decorator ---
    @observe(name="rag_service_ingest_document")
    # --- Langfuse Integration End ---
    # --- MODIFIED: ingest_document to add to FAISS and persist ---
    def ingest_document(self, doc: ChunkedDocument):
        """
        Ingests a chunked document into the RAG service (in-memory doc_store,
        FAISS index, and JSONL file).
        """
        logger.debug("Ingesting document '%s' (source_type=%s) into RAG service", doc.doc_id, doc.source_type)

        # Add to in-memory doc_store
        self.doc_store.append(doc)

        # Add embedding to FAISS index
        if self.faiss_index:
            # Check embedding dimension consistency before adding
            if self.faiss_index.d != len(doc.embedding):
                logger.error("Embedding dimension mismatch for doc '%s'. FAISS index expects %d, got %d. Skipping FAISS add.", doc.doc_id, self.faiss_index.d, len(doc.embedding))
            else:
                self.faiss_index.add(np.array([doc.embedding]).astype('float32'))
                # Persist FAISS index after each ingestion (for simplicity in dev).
                # In prod, this would be batched or handled by a dedicated indexing service.
                faiss.write_index(self.faiss_index, self.faiss_index_path)
                logger.debug("Added embedding for '%s' to FAISS index and saved.", doc.doc_id)
        else:
            logger.warning("FAISS index not initialized. Document '%s' only added to doc_store and JSONL. Rebuilding index might be needed.", doc.doc_id)

        # Persist to JSONL file
        self._append_to_file(doc)
    # --- END MODIFIED ---

# Example of how this service might be used:
if __name__ == "__main__":
    # Ensure LLMInferenceService is ready (Ollama server and nomic-embed-text pulled)
    from services.llm_inference import LLMInferenceService # Moved import here to avoid circular dependencies if run directly

    # Initialize LLM Inference Client
    llm_inf_client = LLMInferenceService()

    # --- MODIFIED RAG Service initialization for file paths ---
    rag_data_file = "demo_rag_knowledge_base.jsonl"
    faiss_index_file = "demo_faiss_index.bin"

    # Clean up previous demo files for a fresh run
    if os.path.exists(rag_data_file):
        os.remove(rag_data_file)
        print(f"Removed existing {rag_data_file} for a clean demo run.")
    if os.path.exists(faiss_index_file):
        os.remove(faiss_index_file)
        print(f"Removed existing {faiss_index_file} for a clean demo run.")

    rag_service = RAGService(llm_inf_client, db_file_path=rag_data_file, faiss_index_path=faiss_index_file)
    print(f"\nInitialized RAGService. JSONL file: {rag_service.db_file_path}, FAISS index: {rag_service.faiss_index_path}")
    print(f"Initial in-memory doc_store size: {len(rag_service.doc_store)}")
    print(f"Initial FAISS index size: {rag_service.faiss_index.ntotal if rag_service.faiss_index else 'N/A'}")
    # --- END MODIFIED RAG Service initialization ---

    print("\n--- Ingesting Documents ---")
    doc1 = ChunkedDocument(
        doc_id="prod_lap_001_desc_chunk",
        content="This high-performance gaming laptop features an Intel i7 processor, 16GB RAM, and a 1TB SSD. Ideal for serious gamers.",
        embedding=llm_inf_client.call_embeddings("high-performance gaming laptop i7"),
        source_type="product_catalog",
        metadata={"product_id": "PROD_LAP_001"}
    )
    doc2 = ChunkedDocument(
        doc_id="policy_returns_general",
        content="Our general return policy states that unused items can be returned within 30 days of purchase, provided they are in original packaging. Some electronics have a 15-day return window.",
        embedding=llm_inf_client.call_embeddings("general return policy for products"),
        source_type="customer_support_policy",
        metadata={"policy_type": "returns"}
    )
    doc3 = ChunkedDocument(
        doc_id="prod_head_002_desc_chunk",
        content="Premium noise-cancelling headphones for immersive audio. Comfortable earcups and 20-hour battery life. Perfect for travel.",
        embedding=llm_inf_client.call_embeddings("best headphones for travel"),
        source_type="product_catalog",
        metadata={"product_id": "PROD_HEAD_002"}
    )

    rag_service.ingest_document(doc1)
    rag_service.ingest_document(doc2)
    rag_service.ingest_document(doc3)

    print(f"\nIn-memory doc_store size after ingestion: {len(rag_service.doc_store)}")
    print(f"FAISS index size after ingestion: {rag_service.faiss_index.ntotal if rag_service.faiss_index else 'N/A'}")

    # Query the knowledge base with real semantic queries
    print("\n--- Querying Knowledge Base (FAISS Semantic Search) ---")

    query_text_1 = "I need information about gaming computers."
    query_embedding_1 = llm_inf_client.call_embeddings(query_text_1)
    results_1 = rag_service.query_knowledge_base(query_embedding_1, query_text_1, top_k=2)
    print(f"\nSemantic Query Results for '{query_text_1}':")
    if results_1:
        for i, res in enumerate(results_1):
            print(f"- {i+1}. Doc ID: {res.doc_id}, Content: '{res.content[:70]}...' (Source: {res.source_type})")
    else:
        print("No results found.")

    query_text_2 = "What is the policy for sending items back?"
    query_embedding_2 = llm_inf_client.call_embeddings(query_text_2)
    results_2 = rag_service.query_knowledge_base(query_embedding_2, query_text_2, top_k=1)
    print(f"\nSemantic Query Results for '{query_text_2}':")
    if results_2:
        for i, res in enumerate(results_2):
            print(f"- {i+1}. Doc ID: {res.doc_id}, Content: '{res.content[:70]}...' (Source: {res.source_type})")
    else:
        print("No results found.")

    query_text_3 = "Comfortable headphones for long flights"
    query_embedding_3 = llm_inf_client.call_embeddings(query_text_3)
    results_3 = rag_service.query_knowledge_base(query_embedding_3, query_text_3, top_k=1)
    print(f"\nSemantic Query Results for '{query_text_3}':")
    if results_3:
        for i, res in enumerate(results_3):
            print(f"- {i+1}. Doc ID: {res.doc_id}, Content: '{res.content[:70]}...' (Source: {res.source_type})")
    else:
        print("No results found.")


    # --- Simulating Service Restart and Loading from File/Index ---
    print("\n--- Simulating Service Restart and Loading from File/Index ---")
    reloaded_rag_service = RAGService(llm_inf_client, db_file_path=rag_data_file, faiss_index_path=faiss_index_file)
    print(f"Reloaded in-memory doc_store size: {len(reloaded_rag_service.doc_store)}")
    print(f"Reloaded FAISS index size: {reloaded_rag_service.faiss_index.ntotal if reloaded_rag_service.faiss_index else 'N/A'}")

    reloaded_results = reloaded_rag_service.query_knowledge_base(query_embedding_1, query_text_1, top_k=1)
    print(f"\nReloaded RAG Query Results for '{query_text_1}':")
    if reloaded_results:
        for i, res in enumerate(reloaded_results):
            print(f"- {i+1}. Doc ID: {res.doc_id}, Content: '{res.content[:70]}...' (Source: {res.source_type})")
    else:
        print("No results found.")
    # --- END NEW: Test loading from file again ---