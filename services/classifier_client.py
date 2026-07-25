# services/classifier_client.py
"""
HTTP client for a separate classifier service - a standalone FastAPI
service hosting encoder-family (BERT-style) classification models,
run and deployed independently of this project.

Two call sites today, both integration hooks only - they log/record the
result, nothing downstream (routing, NLG tone, agent behaviour) reacts to
sentiment yet:
- services/orchestrator.py: live sentiment of each customer message.
- services/data_pipeline.py: product-review sentiment during ingestion,
  replacing what used to be a hardcoded keyword-match mock.

Fails soft everywhere: an unreachable classifier (e.g. not started this
session), a timeout, or an unparseable response all degrade to
SENTIMENT_UNKNOWN instead of raising, so this service being off never
breaks a chat turn or an ingestion run - same philosophy as every
services/llm_inference.py call.

Logging: __init__ probes and logs reachability once, at construction time
(i.e. at container/process startup - see api/dependencies.py's
warm_up_services(), which constructs this eagerly) so the wiring status is
visible in `docker logs` without waiting for a real request. Every
classify_sentiment() call then logs its own outcome at INFO - label,
stars, score, and round-trip latency - so a run of requests shows exactly
whether the classifier is actually being used, not just whether it failed.
"""
import logging
import os
import time

import requests
from dotenv import load_dotenv

from common.models import SentimentResult

load_dotenv()

logger = logging.getLogger(__name__)

SENTIMENT_UNKNOWN = SentimentResult(label="unknown", stars=0, score=0.0, raw_label="classifier_unavailable")


class ClassifierClient:
    def __init__(self):
        self.base_url = os.getenv("CLASSIFIER_API_BASE_URL", "http://localhost:8100")
        self.timeout_seconds = float(os.getenv("CLASSIFIER_TIMEOUT_SECONDS", "5"))

        # One clear line at startup - the whole point being: don't make
        # someone infer classifier wiring from the absence of a warning
        # three requests later. See is_reachable() for what this checks.
        if self.is_reachable():
            logger.info(
                "ClassifierClient: sentiment service reachable at %s (timeout=%.1fs)",
                self.base_url, self.timeout_seconds,
            )
        else:
            logger.warning(
                "ClassifierClient: sentiment service NOT reachable at %s - "
                "classify_sentiment() will return label='unknown' until this is fixed. "
                "Start it separately (shopassist-model's own docker compose) and confirm "
                "CLASSIFIER_API_BASE_URL points at it.",
                self.base_url,
            )

    def is_reachable(self, timeout: float = 3.0) -> bool:
        """Cheap GET /health probe - same check used at startup (above) and by
        GET /api/v1/health (api/routers/health.py) to report classifier_reachable."""
        url = f"{self.base_url.rstrip('/')}/health"
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            logger.debug("is_reachable: %s -> %s", url, response.json())
            return True
        except requests.exceptions.RequestException as e:
            logger.debug("is_reachable: %s unreachable: %s", url, e)
            return False

    def classify_sentiment(self, text: str) -> SentimentResult:
        """Sentiment of `text` via the classifier service's POST /v1/classify/sentiment.

        Mirrors the classifier service's own SentimentRequest/SentimentResponse
        schema exactly. Never raises - see module docstring.
        """
        if not text or not text.strip():
            return SENTIMENT_UNKNOWN

        url = f"{self.base_url.rstrip('/')}/v1/classify/sentiment"
        start = time.perf_counter()
        try:
            response = requests.post(url, json={"text": text}, timeout=self.timeout_seconds)
            response.raise_for_status()
            result = SentimentResult.model_validate(response.json())
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "classify_sentiment: label=%s stars=%d score=%.3f raw_label=%r (%.1fms) text=%r",
                result.label, result.stars, result.score, result.raw_label, latency_ms, text,
            )
            return result
        except requests.exceptions.RequestException as e:
            logger.warning("classify_sentiment: classifier unreachable (%s): %s", url, e)
            return SENTIMENT_UNKNOWN
        except ValueError as e:  # bad JSON, or a response shape that fails SentimentResult validation
            logger.warning("classify_sentiment: unexpected response from classifier: %s", e)
            return SENTIMENT_UNKNOWN
