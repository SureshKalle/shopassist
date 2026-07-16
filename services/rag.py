# services/rag.py
import json
from abc import ABC, abstractmethod
from typing import List, Dict

from sqlalchemy import create_engine, text

from common.models import ChunkedDocument
from services.llm_inference import MockLLMInferenceService # To get embeddings


def _to_vector_literal(embedding: List[float]) -> str:
    """Pgvector's own text literal format ("[0.1,0.2,...]"), cast in SQL via
    CAST(:param AS vector). Explicit string formatting + CAST rather than
    relying on pgvector-python's register_vector: tested against this
    project's actual SQLAlchemy engine + connection-pool setup and found
    unreliable there (psycopg2 kept sending plain Python lists as bare
    Postgres arrays, not vector literals, so a real similarity query failed
    with "operator does not exist: vector <=> numeric[]") — this sidesteps
    the adapter entirely, at the cost of one small helper.
    """
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _from_vector_literal(value: str) -> List[float]:
    """Inverse of _to_vector_literal — pgvector's default text output for a
    `vector` column, since nothing above registers a binary typecaster.
    """
    return [float(x) for x in value.strip("[]").split(",")]


class BaseRAGService(ABC):
    """Shared interface for the RAG backends below — mirrors
    services/agents/base_agent.py::BaseAgent's role for agents. Callers
    (services/data_pipeline.py, services/agents/*.py) only ever depend on
    this shape, not on which concrete backend is active.
    """

    @abstractmethod
    def collection_size(self) -> int:
        ...

    @abstractmethod
    def query_knowledge_base(self, query_embedding: List[float], query_text: str, top_k: int = 1) -> List[ChunkedDocument]:
        ...

    @abstractmethod
    def ingest_document(self, doc: ChunkedDocument) -> None:
        ...


class MockRAGService(BaseRAGService):
    """
    In-memory, keyword-matched stand-in for the SQLite/no-Postgres path —
    SQLite has no vector extension, and this project's own convention
    already treats SQLite as disposable local-dev data (see
    shopassist-database's docs/database-design.md), so it gets this mock
    instead of a from-scratch similarity search. `PgVectorRAGService` below
    is the real implementation, used whenever `DATABASE_URL` is Postgres.
    """
    def __init__(self, llm_inference_client: MockLLMInferenceService):
        self.vector_db: Dict[str, ChunkedDocument] = {} # {doc_id: ChunkedDocument}
        self.rag_query_cache: Dict[str, List[ChunkedDocument]] = {}
        self.llm_inference_client = llm_inference_client

    def collection_size(self) -> int:
        """Number of chunks currently ingested — used by the /health endpoint."""
        return len(self.vector_db)

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

    def ingest_document(self, doc: ChunkedDocument) -> None:
        """
        Ingests a chunked document into the RAG vector database.
        In a real system, this would write to a persistent vector store.
        """
        print(f"  [Mock RAG] Ingesting document '{doc.doc_id}' into vector DB...")
        self.vector_db[doc.doc_id] = doc


class PgVectorRAGService(BaseRAGService):
    """Real RAG backend: pgvector-backed semantic search against
    shopassist-database's `document_chunks` table (see that repo's
    docs/database-design.md for the schema/index rationale).

    Plain SQLAlchemy Core + text() throughout, matching
    services/ecommerce_client.py's style — no ORM models anywhere in this
    repo. Embeddings cross the SQL boundary as pgvector's own text literal
    format (`_to_vector_literal`/`_from_vector_literal` above), cast
    explicitly via `CAST(:param AS vector)` — see those functions' docstrings
    for why, over the pgvector-python package's usual adapter registration.
    """

    def __init__(self, llm_inference_client: MockLLMInferenceService, database_url: str):
        self.llm_inference_client = llm_inference_client
        self.engine = create_engine(database_url)

    def collection_size(self) -> int:
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT count(*) FROM document_chunks")).scalar_one()

    def query_knowledge_base(self, query_embedding: List[float], query_text: str, top_k: int = 1) -> List[ChunkedDocument]:
        print(f"  [PgVector RAG] Querying document_chunks for '{query_text}'...")
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT doc_id, content, embedding, source_type, metadata
                    FROM document_chunks
                    ORDER BY embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :top_k
                """),
                {"query_embedding": _to_vector_literal(query_embedding), "top_k": top_k},
            ).mappings().all()

        return [
            ChunkedDocument(
                doc_id=row["doc_id"],
                content=row["content"],
                embedding=_from_vector_literal(row["embedding"]),
                source_type=row["source_type"],
                metadata=row["metadata"],
            )
            for row in rows
        ]

    def ingest_document(self, doc: ChunkedDocument) -> None:
        print(f"  [PgVector RAG] Ingesting document '{doc.doc_id}' into document_chunks...")
        with self.engine.begin() as conn:
            conn.execute(
                # CAST(...), not :name::type — SQLAlchemy's text() bind-
                # parameter parser doesn't reliably recognize a bind name
                # immediately followed by a Postgres `::` cast (confirmed by
                # testing: it left ":metadata::jsonb" completely unbound
                # while every other parameter converted correctly). CAST()
                # sidesteps the ambiguity entirely for both casts here.
                text("""
                    INSERT INTO document_chunks (doc_id, content, embedding, source_type, metadata)
                    VALUES (:doc_id, :content, CAST(:embedding AS vector), :source_type, CAST(:metadata AS jsonb))
                    ON CONFLICT (doc_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        source_type = EXCLUDED.source_type,
                        metadata = EXCLUDED.metadata,
                        updated_at = now()
                """),
                {
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "embedding": _to_vector_literal(doc.embedding),
                    "source_type": doc.source_type,
                    "metadata": json.dumps(doc.metadata),
                },
            )


# Example of how this service might be used:
if __name__ == "__main__":
    llm_inf_client = MockLLMInferenceService()
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
