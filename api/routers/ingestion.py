# api/routers/ingestion.py
"""
Data ingestion endpoints — feed the RAG knowledge base.

POST /api/v1/ingest/policies         → ingest policy documents (JSON body)
POST /api/v1/ingest/products         → ingest product catalogue records (JSON body)
POST /api/v1/ingest/twitter-csv      → ingest a Twitter-support-style CSV (file upload)

These wrap `DataIngestionPipeline` (services/data_pipeline.py) — no ingestion
logic lives in the API layer itself, keeping the pipeline reusable from the
CLI simulation, tests, or any other caller.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import get_data_pipeline, get_rag_service
from api.schemas import IngestPoliciesRequest, IngestProductsRequest, IngestResponse
from common.models import RawProductRecord
from services.data_pipeline import DataIngestionPipeline
from services.rag import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


@router.post("/policies", response_model=IngestResponse)
def ingest_policies(
    request: IngestPoliciesRequest,
    pipeline: DataIngestionPipeline = Depends(get_data_pipeline),
    rag_service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    """Ingest one or more policy documents into the RAG knowledge base."""
    policies = [p.model_dump() for p in request.policies]
    pipeline.ingest_policy_documents(policies)
    return IngestResponse(
        ingested_count=len(policies),
        rag_collection_size=rag_service.collection_size(),
        message=f"Ingested {len(policies)} policy document(s).",
    )


@router.post("/products", response_model=IngestResponse)
def ingest_products(
    request: IngestProductsRequest,
    pipeline: DataIngestionPipeline = Depends(get_data_pipeline),
    rag_service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    """Ingest one or more product catalogue records into the RAG knowledge base."""
    raw_products = [
        RawProductRecord(
            product_id=p.product_id,
            raw_description=p.raw_description,
            specs=p.specs,
            reviews=p.reviews,
            price=p.price,
        )
        for p in request.products
    ]
    cleaned = pipeline.ingest_product_catalog(raw_products)
    return IngestResponse(
        ingested_count=len(cleaned),
        rag_collection_size=rag_service.collection_size(),
        message=f"Ingested {len(cleaned)} product record(s).",
    )


@router.post("/twitter-csv", response_model=IngestResponse)
async def ingest_twitter_csv(
    file: UploadFile = File(..., description="CSV with columns: tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id"),
    max_conversations: int | None = None,
    pipeline: DataIngestionPipeline = Depends(get_data_pipeline),
    rag_service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    """
    Ingest a "Customer Support on Twitter"-style CSV file (uploaded directly)
    into the RAG knowledge base. See `DataIngestionPipeline.ingest_twitter_support_csv`
    for the inbound/outbound tweet-pairing logic.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        cleaned = pipeline.ingest_twitter_support_csv(tmp_path, max_conversations=max_conversations)
    except Exception as exc:
        logger.error("[API] Twitter CSV ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        ingested_count=len(cleaned),
        rag_collection_size=rag_service.collection_size(),
        message=f"Ingested {len(cleaned)} conversation(s) from '{file.filename}'.",
    )
