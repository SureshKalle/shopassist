# services/rag.py
"""
RAG (Retrieval-Augmented Generation) Service — LangChain edition.

Uses:
  - `langchain_core.embeddings.Embeddings`  → pluggable embedding provider
  - `langchain_chroma.Chroma`               → vectorstore wrapper over ChromaDB
  - `.as_retriever(search_kwargs=...)`      → standard LangChain retriever interface

This means any other LangChain-compatible component (a `RetrievalQA` chain,
a LangChain agent's retriever tool, LCEL `RunnableParallel` fan-out, etc.)
can plug straight into `RAGService.as_retriever()` with zero adapter code.

Embeddings provider:
  No embeddings API key is configured yet (Anthropic doesn't offer one).
  `_DeterministicEmbeddings` is a zero-dependency stand-in implementing the
  full `Embeddings` interface so the rest of the LangChain stack (Chroma,
  retrievers, chains) works unmodified today. Swapping in a real provider
  later is a one-line change in `get_embeddings_model()`:

      from langchain_openai import OpenAIEmbeddings
      return OpenAIEmbeddings(model="text-embedding-3-small")

      # or Voyage AI (recommended by Anthropic for Claude-based RAG):
      from langchain_voyageai import VoyageAIEmbeddings
      return VoyageAIEmbeddings(model="voyage-3")
"""

import logging
from typing import Optional, TYPE_CHECKING, Union, cast

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from common.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PROVIDER,
    FAISS_PERSIST_DIR,
    RAG_SIMILARITY_THRESHOLD,
    RAG_TOP_K,
    VECTOR_STORE,
)
from common.models import ChunkedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embeddings provider
# ---------------------------------------------------------------------------
class _DeterministicEmbeddings(Embeddings):
    """
    Zero-dependency offline fallback implementing LangChain's `Embeddings`
    interface via a 64-dim character-frequency histogram. Only used if
    EMBEDDING_PROVIDER="deterministic" or if loading the real model fails
    (e.g. no internet to download weights on first run). Not suitable for
    RAGAS evaluation — retrieval quality will be poor.
    """

    DIM = 64

    def _vectorize(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for ch in text:
            vec[ord(ch) % self.DIM] += 1.0
        norm = max(sum(v * v for v in vec) ** 0.5, 1e-9)
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


_embeddings_singleton: Optional[Embeddings] = None


def get_embeddings_model() -> Embeddings:
    """
    Factory for the embeddings model used across the platform (RAG ingestion
    + queries, and `LLMInferenceService.call_embeddings` for backward compat).

    Default: `sentence-transformers/all-MiniLM-L6-v2` running locally via
    `langchain_huggingface.HuggingFaceEmbeddings` — matches the Tech Stack
    spec's "Embedding Models" row (all-MiniLM-L6-v2 / BGE), needs no API key,
    and produces retrieval quality that's actually meaningful for RAGAS
    Context Recall/Precision scoring. Swap the model name in .env to use BGE
    (e.g. "BAAI/bge-small-en-v1.5") without touching this code.
    """
    global _embeddings_singleton
    if _embeddings_singleton is not None:
        return _embeddings_singleton

    if EMBEDDING_PROVIDER == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            _embeddings_singleton = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            logger.info(
                "[RAG] Loaded HuggingFace embeddings model '%s'.", EMBEDDING_MODEL_NAME
            )
            return _embeddings_singleton
        except Exception as exc:
            logger.error(
                "[RAG] Failed to load HuggingFace embeddings ('%s'): %s. "
                "Falling back to deterministic embeddings — retrieval quality "
                "will be degraded.", EMBEDDING_MODEL_NAME, exc,
            )

    _embeddings_singleton = _DeterministicEmbeddings()
    logger.warning("[RAG] Using deterministic (non-semantic) embeddings fallback.")
    return _embeddings_singleton


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------
class RAGService:
    """
    LangChain-powered RAG Service backed by a persistent Chroma collection.

    Public API:
        ingest_document(doc)                  → add/update a ChunkedDocument
        query_knowledge_base(...)             → retrieve top-K ChunkedDocuments
        as_retriever(search_kwargs=None)       → standard LangChain retriever
        delete_document(doc_id)               → remove a document
        collection_size()                     → number of stored chunks
    """

    def __init__(self, llm_inference_client=None):
        # llm_inference_client kept as an optional positional arg for
        # backward compatibility with existing call sites; embeddings are
        # now resolved via get_embeddings_model() rather than the LLM service.
        self.llm_inference_client = llm_inference_client
        self._embeddings = get_embeddings_model()
        self._backend = VECTOR_STORE.lower()

        self._vectorstore: Union[Chroma, "FAISS"]
        if self._backend == "faiss":
            self._vectorstore = self._init_faiss()
        else:
            self._vectorstore = self._init_chroma()

        logger.info(
            "[RAG] Vectorstore ready | backend=%s | %d documents.",
            self._backend, self.collection_size(),
        )

    def _init_chroma(self) -> Chroma:
        logger.info(
            "[RAG] Initialising Chroma vectorstore | dir='%s' collection='%s'",
            CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
        )
        return Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _init_faiss(self) -> "FAISS":
        """
        FAISS backend (per Tech Stack: "ChromaDB / FAISS"). Loads an existing
        index from FAISS_PERSIST_DIR if present, otherwise starts empty and
        persists on first ingest.

        Note: FAISS (via langchain_community) does not support server-side
        metadata `where`-filtering as richly as Chroma — `source_filter` in
        `query_knowledge_base` is applied as a post-filter here instead.
        """
        import os

        from langchain_community.vectorstores import FAISS

        if os.path.isdir(FAISS_PERSIST_DIR) and os.listdir(FAISS_PERSIST_DIR):
            logger.info("[RAG] Loading existing FAISS index from '%s'.", FAISS_PERSIST_DIR)
            return FAISS.load_local(
                FAISS_PERSIST_DIR, self._embeddings, allow_dangerous_deserialization=True
            )

        logger.info("[RAG] No existing FAISS index found — starting empty (dir='%s').", FAISS_PERSIST_DIR)
        # FAISS requires at least one vector to initialise; seed with a placeholder
        # that is immediately distinguishable and never returned (filtered by score).
        return FAISS.from_texts(["__faiss_init_placeholder__"], self._embeddings)

    def _persist_faiss(self) -> None:
        if self._backend == "faiss":
            cast("FAISS", self._vectorstore).save_local(FAISS_PERSIST_DIR)

    # ──────────────────────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_document(self, doc: ChunkedDocument) -> None:
        """Add or update (upsert) a document in the Chroma vectorstore."""
        logger.debug("[RAG] Ingesting doc_id='%s' source='%s'", doc.doc_id, doc.source_type)

        lc_doc = Document(
            page_content=doc.content,
            metadata={
                "doc_id": doc.doc_id,
                "source_type": doc.source_type,
                **{k: str(v) for k, v in doc.metadata.items()},
            },
        )
        # Chroma's add_documents with explicit ids acts as an upsert.
        # FAISS's add_documents appends; true upsert-by-id isn't natively
        # supported, so re-ingesting the same doc_id with FAISS will create
        # a duplicate vector — acceptable for this project's batch-ingest
        # pattern (fresh collection per run) but worth knowing if you add
        # incremental re-ingestion later.
        self._vectorstore.add_documents(documents=[lc_doc], ids=[doc.doc_id])
        self._persist_faiss()

    def ingest_documents_batch(self, docs: list[ChunkedDocument]) -> None:
        """Batch-ingest multiple documents in a single Chroma call (more efficient)."""
        if not docs:
            return
        lc_docs = [
            Document(
                page_content=d.content,
                metadata={
                    "doc_id": d.doc_id,
                    "source_type": d.source_type,
                    **{k: str(v) for k, v in d.metadata.items()},
                },
            )
            for d in docs
        ]
        ids = [d.doc_id for d in docs]
        self._vectorstore.add_documents(documents=lc_docs, ids=ids)
        self._persist_faiss()
        logger.info("[RAG] Batch-ingested %d document(s).", len(docs))

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def query_knowledge_base(
        self,
        query_embedding: Optional[list[float]],   # kept for backward-compat signature; unused
        query_text: str,
        top_k: int = RAG_TOP_K,
        source_filter: Optional[str] = None,
    ) -> list[ChunkedDocument]:
        """
        Retrieve the most relevant documents for a query using Chroma's
        similarity search with relevance scores.

        Note: `query_embedding` is accepted for backward compatibility with
        the previous signature (agents pre-compute embeddings), but Chroma's
        `similarity_search_with_relevance_scores` re-embeds the query text
        internally via the configured embedding function. This keeps a single
        source of truth for the embedding space.
        """
        logger.info("[RAG] Querying for: %.80s (top_k=%d, filter=%s)", query_text, top_k, source_filter)

        where = {"source_type": source_filter} if (source_filter and self._backend == "chroma") else None

        try:
            if self._backend == "faiss" and source_filter:
                # FAISS: over-fetch then post-filter by metadata (no native where-filter)
                raw_results = self._vectorstore.similarity_search_with_relevance_scores(
                    query_text, k=max(top_k * 4, 10),
                )
                results = [
                    (doc, score) for doc, score in raw_results
                    if doc.metadata.get("source_type") == source_filter
                ][:top_k]
            else:
                results = self._vectorstore.similarity_search_with_relevance_scores(
                    query_text, k=top_k, filter=where,
                )
        except Exception as exc:
            logger.error("[RAG] Vectorstore query failed: %s", exc)
            return []

        docs: list[ChunkedDocument] = []
        for lc_doc, score in results:
            if lc_doc.page_content == "__faiss_init_placeholder__":
                continue
            if score < RAG_SIMILARITY_THRESHOLD:
                logger.debug(
                    "[RAG] Skipping doc (score=%.3f < threshold=%.3f)", score, RAG_SIMILARITY_THRESHOLD
                )
                continue
            meta = dict(lc_doc.metadata)
            source_type = meta.pop("source_type", "unknown")
            docs.append(
                ChunkedDocument(
                    doc_id=meta.get("doc_id", ""),
                    content=lc_doc.page_content,
                    embedding=[],
                    source_type=source_type,
                    metadata={**meta, "similarity_score": round(score, 4)},
                )
            )

        logger.info("[RAG] Returned %d document(s).", len(docs))
        return docs

    def as_retriever(
        self, search_kwargs: Optional[dict] = None
    ) -> VectorStoreRetriever:
        """
        Return a standard LangChain retriever for use in LCEL chains,
        LangChain agents (as a retriever tool), or `create_retrieval_chain`.

        Example:
            retriever = rag_service.as_retriever(
                search_kwargs={"k": 3, "filter": {"source_type": "product_catalog"}}
            )
            docs = retriever.invoke("gaming laptop recommendation")

        Note (FAISS backend only): the internal placeholder vector used to
        initialise an empty FAISS index may appear in raw retriever results
        until enough real documents are ingested to outrank it. This is
        filtered out automatically in `query_knowledge_base()`, but callers
        using `as_retriever()` directly (e.g. inside a LangChain agent tool)
        should be aware of it for a brand-new, freshly-created FAISS index.
        """
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs or {"k": RAG_TOP_K})

    # ──────────────────────────────────────────────────────────────────────────
    # Management helpers
    # ──────────────────────────────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> None:
        self._vectorstore.delete(ids=[doc_id])
        self._persist_faiss()
        logger.info("[RAG] Deleted doc_id='%s'.", doc_id)

    def collection_size(self) -> int:
        try:
            if self._backend == "faiss":
                # Subtract the placeholder seed vector if the index is otherwise empty
                return cast("FAISS", self._vectorstore).index.ntotal
            return cast(Chroma, self._vectorstore)._collection.count()
        except Exception:
            return 0


# ── Backwards-compatible alias ────────────────────────────────────────────────
MockRAGService = RAGService


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rag = RAGService()
    sample = ChunkedDocument(
        doc_id="prod_lap_001_test",
        content="High-performance gaming laptop with i7 processor, 16GB RAM, 1TB SSD.",
        embedding=[],
        source_type="product_catalog",
        metadata={"product_id": "PROD_LAP_001"},
    )
    rag.ingest_document(sample)
    print(f"Collection size: {rag.collection_size()}")

    results = rag.query_knowledge_base(None, "gaming laptop recommendation")
    for r in results:
        print(f"  → score={r.metadata.get('similarity_score')} | {r.content[:60]}…")

    # LangChain retriever interface demo
    retriever = rag.as_retriever()
    docs = retriever.invoke("gaming laptop")
    print(f"Retriever returned {len(docs)} LangChain Document(s).")
