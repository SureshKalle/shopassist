# services/data_pipeline.py
"""
DataIngestionPipeline — LangChain edition.

Chunking now uses `RecursiveCharacterTextSplitter` (LangChain's standard,
battle-tested splitter) instead of a hand-rolled sentence splitter. It tries
paragraph → sentence → word boundaries in order, which handles messy raw
text (HTML remnants, run-on reviews, etc.) more robustly than a single regex.

Pipeline stages (unchanged conceptually):
  Raw data → clean & normalise → PII mask → chunk (LangChain splitter) →
  RAGService.ingest_documents_batch() (embeds + upserts into Chroma)
"""

import csv
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.models import (
    ChunkedDocument,
    CleanedCustomerConversation,
    CleanedProductRecord,
    RawCustomerConversation,
    RawProductRecord,
)
from services.pii_masker import PIIMasker
from services.rag import RAGService

logger = logging.getLogger(__name__)

_MAX_CHUNK_CHARS = 800
_CHUNK_OVERLAP = 80

# Shared splitter instance — paragraph → sentence → word → char fallback order
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_MAX_CHUNK_CHARS,
    chunk_overlap=_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
)


class DataIngestionPipeline:
    """
    Handles data collection, cleaning, PII masking, chunking (via LangChain's
    RecursiveCharacterTextSplitter), and RAG ingestion (via RAGService, which
    wraps a LangChain Chroma vectorstore).
    """

    def __init__(self, pii_masker: PIIMasker, llm_inference_client, rag_service: RAGService):
        self.pii_masker = pii_masker
        self.llm_inference_client = llm_inference_client  # kept for backward-compat signature
        self.rag_service = rag_service

    # ──────────────────────────────────────────────────────────────────────────
    # Customer conversations
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_customer_conversations(
        self, raw_conversations: list[RawCustomerConversation]
    ) -> list[CleanedCustomerConversation]:
        logger.info("[Pipeline] Ingesting %d customer conversation(s).", len(raw_conversations))
        cleaned: list[CleanedCustomerConversation] = []
        docs_to_ingest: list[ChunkedDocument] = []

        for raw in raw_conversations:
            logger.debug("[Pipeline] Processing conversation id=%s", raw.id)

            text = self._clean_text(raw.text)
            masked_data = self.pii_masker.mask_text(
                text, session_id=raw.id, user_id=raw.metadata.get("user_id")
            )
            masked_text = masked_data.masked_text

            cleaned.append(
                CleanedCustomerConversation(
                    id=raw.id,
                    cleaned_text=masked_text,
                    tokens=masked_text.split(),
                    metadata={**raw.metadata, "original_text_hash": masked_data.original_text_hash},
                )
            )

            if len(masked_text) < 15:
                logger.debug("[Pipeline] Skipping short/empty conversation %s.", raw.id)
                continue

            for i, chunk_content in enumerate(_splitter.split_text(masked_text)):
                docs_to_ingest.append(
                    ChunkedDocument(
                        doc_id=f"conv_chunk_{raw.id}_{i}",
                        content=chunk_content,
                        embedding=[],  # embedding is computed inside RAGService/Chroma now
                        source_type="customer_support_conversation",
                        metadata={"conv_id": raw.id, "chunk_idx": i},
                    )
                )

        self.rag_service.ingest_documents_batch(docs_to_ingest)
        logger.info("[Pipeline] Conversation ingestion complete. %d record(s) cleaned.", len(cleaned))
        return cleaned

    # ──────────────────────────────────────────────────────────────────────────
    # Twitter customer-support CSV ingestion (Data Strategy source)
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_twitter_support_csv(
        self,
        csv_path: str | Path,
        max_conversations: Optional[int] = None,
    ) -> list[CleanedCustomerConversation]:
        """
        Parse a "Customer Support on Twitter"-style CSV
        (columns: tweet_id, author_id, inbound, created_at, text,
        response_tweet_id, in_response_to_tweet_id) into paired
        customer↔support conversation turns, then run them through the
        standard `ingest_customer_conversations` cleaning/masking/chunking
        pipeline.

        Pairing logic:
          - `inbound == "True"`  → customer tweet
          - `inbound == "False"` → brand/support reply
          - A conversation turn is formed by following `response_tweet_id`
            from an inbound tweet to its outbound reply. Tweets with no
            reply (dangling threads) are still ingested as single-turn
            customer messages, since they carry useful intent signal for
            the router LLM even without a resolution.

        Args:
            csv_path: Path to the CSV file (e.g. the sample Twitter dataset).
            max_conversations: Optional cap, useful for quick local testing
                                on large CSVs without ingesting everything.

        Returns:
            Cleaned conversation records (same shape as
            `ingest_customer_conversations`, so callers can treat both
            sources uniformly).
        """
        csv_path = Path(csv_path)
        logger.info("[Pipeline] Loading Twitter support CSV from '%s'.", csv_path)

        rows_by_id: dict[str, dict] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_by_id[row["tweet_id"]] = row

        logger.info("[Pipeline] Loaded %d raw tweet(s).", len(rows_by_id))

        # Build paired conversations: inbound tweet + its first outbound reply
        raw_conversations: list[RawCustomerConversation] = []
        seen_as_reply: set[str] = set()

        for tweet_id, row in rows_by_id.items():
            is_inbound = row.get("inbound", "").strip().lower() == "true"
            if not is_inbound or tweet_id in seen_as_reply:
                continue

            customer_text = row["text"]
            author_id = row["author_id"]

            # Follow the reply chain (first response only, for a simple 1-turn pair)
            reply_ids = [r for r in row.get("response_tweet_id", "").split(",") if r]
            support_text = None
            brand = None
            for reply_id in reply_ids:
                reply_row = rows_by_id.get(reply_id)
                if reply_row and reply_row.get("inbound", "").strip().lower() == "false":
                    support_text = reply_row["text"]
                    brand = reply_row["author_id"]
                    seen_as_reply.add(reply_id)
                    break

            conversation_text = (
                f"Customer: {customer_text}\nSupport ({brand}): {support_text}"
                if support_text
                else f"Customer: {customer_text}"
            )

            raw_conversations.append(
                RawCustomerConversation(
                    id=f"twitter_{tweet_id}",
                    text=conversation_text,
                    metadata={
                        "source": "twitter",
                        "user_id": author_id,
                        "brand": brand or "unknown",
                        "created_at": row.get("created_at", ""),
                        "resolved": bool(support_text),
                    },
                )
            )

            if max_conversations and len(raw_conversations) >= max_conversations:
                break

        logger.info(
            "[Pipeline] Paired into %d conversation(s) (%d with a support reply).",
            len(raw_conversations),
            sum(1 for c in raw_conversations if c.metadata.get("resolved")),
        )

        return self.ingest_customer_conversations(raw_conversations)

    # ──────────────────────────────────────────────────────────────────────────
    # Product catalogue
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_product_catalog(
        self, raw_products: list[RawProductRecord]
    ) -> list[CleanedProductRecord]:
        logger.info("[Pipeline] Ingesting %d product record(s).", len(raw_products))
        cleaned: list[CleanedProductRecord] = []
        docs_to_ingest: list[ChunkedDocument] = []

        for raw in raw_products:
            logger.debug("[Pipeline] Processing product id=%s", raw.product_id)

            clean_description = self._clean_text(raw.raw_description, lowercase=False)
            normalized_price = self._parse_price(raw.price)
            structured_specs = {k.lower().replace(" ", "_"): v for k, v in raw.specs.items()}

            seen_reviews: set[str] = set()
            sentiment_analyzed_reviews: list[dict] = []
            for review_text in raw.reviews:
                normalised_review = self._clean_text(review_text, lowercase=False)
                if normalised_review in seen_reviews:
                    continue
                seen_reviews.add(normalised_review)

                masked_review = self.pii_masker.mask_text(normalised_review)
                sentiment = self._mock_sentiment(normalised_review)
                sentiment_analyzed_reviews.append(
                    {
                        "text": masked_review.masked_text,
                        "sentiment": sentiment,
                        "original_hash": masked_review.original_text_hash,
                    }
                )

            cleaned.append(
                CleanedProductRecord(
                    product_id=raw.product_id,
                    clean_description=clean_description,
                    structured_specs=structured_specs,
                    sentiment_analyzed_reviews=sentiment_analyzed_reviews,
                    normalized_price=normalized_price,
                    metadata={"original_price_str": raw.price},
                )
            )

            positive_reviews = " ".join(r["text"] for r in sentiment_analyzed_reviews if r["sentiment"] == "positive")
            negative_reviews = " ".join(r["text"] for r in sentiment_analyzed_reviews if r["sentiment"] == "negative")
            specs_text = ", ".join(f"{k}: {v}" for k, v in structured_specs.items())

            rag_content = (
                f"Product: {clean_description}. Price: ${normalized_price:.2f}. "
                f"Specifications: {specs_text}. "
                + (f"Positive reviews: {positive_reviews}. " if positive_reviews else "")
                + (f"Concerns: {negative_reviews}." if negative_reviews else "")
            )

            for i, chunk_content in enumerate(_splitter.split_text(rag_content)):
                docs_to_ingest.append(
                    ChunkedDocument(
                        doc_id=f"prod_chunk_{raw.product_id}_{i}",
                        content=chunk_content,
                        embedding=[],
                        source_type="product_catalog",
                        metadata={"product_id": raw.product_id, "chunk_idx": i},
                    )
                )

        self.rag_service.ingest_documents_batch(docs_to_ingest)
        logger.info("[Pipeline] Product ingestion complete. %d record(s) cleaned.", len(cleaned))
        return cleaned

    # ──────────────────────────────────────────────────────────────────────────
    # Policy documents
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_policy_documents(self, policies: list[dict]) -> None:
        """
        Ingest plain-text policy documents into the RAG knowledge base.
        Each item: {"id": str, "title": str, "content": str}
        """
        logger.info("[Pipeline] Ingesting %d policy document(s).", len(policies))
        docs_to_ingest: list[ChunkedDocument] = []

        for pol in policies:
            for i, chunk_content in enumerate(_splitter.split_text(pol["content"])):
                docs_to_ingest.append(
                    ChunkedDocument(
                        doc_id=f"{pol['id']}_chunk_{i}",
                        content=chunk_content,
                        embedding=[],
                        source_type="customer_support_policy",
                        metadata={"policy_id": pol["id"], "title": pol.get("title", ""), "chunk_idx": i},
                    )
                )

        self.rag_service.ingest_documents_batch(docs_to_ingest)
        logger.info("[Pipeline] Policy ingestion complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # Synthetic query generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_synthetic_queries(self, base_queries: list[str]) -> list[str]:
        """
        Generate paraphrased variants of seed queries for LLM fine-tuning.
        Template-based augmentation stand-in; swap for an LCEL paraphrase
        chain (prompt | llm | StrOutputParser) once a fine-tuning pipeline exists.
        """
        logger.info("[Pipeline] Generating synthetic queries from %d seed(s).", len(base_queries))
        synthetic: list[str] = []
        templates = [
            lambda q: q,
            lambda q: q.lower().replace("?", ""),
            lambda q: f"Hi, {q[0].lower()}{q[1:]}",
            lambda q: f"I was wondering, {q[0].lower()}{q[1:]}",
            lambda q: f"Can you help me? {q}",
        ]
        for query in base_queries:
            for t in templates:
                variant = t(query).strip()
                if variant and variant not in synthetic:
                    synthetic.append(variant)

        synthetic = [s for s in synthetic if len(s.split()) >= 3]
        logger.info("[Pipeline] Generated %d synthetic query variant(s).", len(synthetic))
        return synthetic

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str, lowercase: bool = True) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        if lowercase:
            text = text.lower()
        return text.strip()

    @staticmethod
    def _parse_price(price_str: str) -> float:
        cleaned = re.sub(r"[^\d.]", "", price_str)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    @staticmethod
    def _mock_sentiment(text: str) -> str:
        positive_words = {"great", "awesome", "excellent", "love", "amazing", "fantastic", "good", "perfect"}
        negative_words = {"bad", "terrible", "awful", "poor", "broken", "worst", "horrible", "disappointing"}
        tokens = set(text.lower().split())
        if tokens & positive_words:
            return "positive"
        if tokens & negative_words:
            return "negative"
        return "neutral"
