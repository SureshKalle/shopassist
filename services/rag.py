# services/rag.py
from typing import List, Dict, Any, Optional
from common.models import ChunkedDocument
from services.llm_inference import LLMInferenceService # To get embeddings

class MockRAGService:
    """
    Simulates the RAG Service.
    In a real system, this would be a microservice interacting with a vector database
    (e.g., Pinecone, Weaviate, ChromaDB) and potentially a search cache (Redis).
    """
    def __init__(self, llm_inference_client: LLMInferenceService):
        self.vector_db: Dict[str, ChunkedDocument] = {} # {doc_id: ChunkedDocument}
        self.rag_query_cache: Dict[str, List[ChunkedDocument]] = {}
        self.llm_inference_client = llm_inference_client

    def query_knowledge_base(self, query_embedding: List[float], query_text: str, top_k: int = 1) -> List[ChunkedDocument]:
        print(f"  [Mock RAG] Querying knowledge base for '{query_text}'...")
        
        # Check cache first
        if query_text in self.rag_query_cache:
            print("  [Mock RAG] Cache hit!")
            return self.rag_query_cache[query_text][:top_k]

        # Simulate retrieval from vector DB (simple keyword match for mock)
        results = []
        for doc_id, doc in self.vector_db.items():
            # In a real system, this would be a vector similarity search
            if query_text.lower() in doc.content.lower():
                results.append(doc)
            elif "return policy" in query_text.lower() and doc.source_type == "customer_support_policy":
                results.append(doc)
            elif "laptop features" in query_text.lower() and doc.source_type == "product_catalog":
                results.append(doc)
            # Add more sophisticated mock matching here if needed

        # Sort by some mock relevance (e.g., length of content match)
        results.sort(key=lambda x: len(x.content) if query_text.lower() in x.content.lower() else 0, reverse=True)

        # Cache results for future queries
        self.rag_query_cache[query_text] = results
        
        return results[:top_k]

    def ingest_document(self, doc: ChunkedDocument):
        """
        Ingests a chunked document into the RAG vector database.
        In a real system, this would write to a persistent vector store.
        """
        print(f"  [Mock RAG] Ingesting document '{doc.doc_id}' into vector DB...")
        self.vector_db[doc.doc_id] = doc

# Example of how this service might be used:
if __name__ == "__main__":
    llm_inf_client = LLMInferenceService()
    rag_service = MockRAGService(llm_inf_client)
    
    # Ingest some mock documents
    doc1 = ChunkedDocument(
        doc_id="prod_lap_001_desc_chunk",
        content="This high-performance laptop features an i7 processor, 16GB RAM, and a 1TB SSD. Ideal for gaming and professional use.",
        embedding=llm_inf_client.call_embeddings("high-performance laptop features"),
        source_type="product_catalog",
        metadata={"product_id": "PROD_LAP_001"}
    )
    doc2 = ChunkedDocument(
        doc_id="policy_returns_general",
        content="Our general return policy states that items can be returned within 30 days of purchase, provided they are in original packaging.",
        embedding=llm_inf_client.call_embeddings("general return policy"),
        source_type="customer_support_policy",
        metadata={"policy_type": "returns"}
    )
    rag_service.ingest_document(doc1)
    rag_service.ingest_document(doc2)

    # Query the knowledge base
    query_text = "What are the features of your high-end gaming laptops?"
    query_embedding = llm_inf_client.call_embeddings(query_text)
    results = rag_service.query_knowledge_base(query_embedding, query_text)
    print(f"\nRAG Query Results for '{query_text}':")
    for res in results:
        print(f"- Doc ID: {res.doc_id}, Content: '{res.content[:50]}...'")
    
    query_text_2 = "What is your return policy?"
    query_embedding_2 = llm_inf_client.call_embeddings(query_text_2)
    results_2 = rag_service.query_knowledge_base(query_embedding_2, query_text_2)
    print(f"\nRAG Query Results for '{query_text_2}':")
    for res in results_2:
        print(f"- Doc ID: {res.doc_id}, Content: '{res.content[:50]}...'")
