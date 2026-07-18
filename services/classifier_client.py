# services/classifier_client.py
"""
HTTP client for shopassist-model's classifier service - a separate FastAPI
service hosting encoder-family (BERT-style) classification models
alongside Ollama's decoder-family generative models (see that repo's
README's "Encoder models" section for why they're split out; its
"Connecting shopassist to this service" section documents this exact
client, base URL default, and env var).

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
"""
import logging
import os

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

    def classify_sentiment(self, text: str) -> SentimentResult:
        """Sentiment of `text` via the classifier service's POST /v1/classify/sentiment.

        Mirrors shopassist-model/classifier/schemas.py's SentimentRequest/
        SentimentResponse exactly. Never raises - see module docstring.
        """
        if not text or not text.strip():
            return SENTIMENT_UNKNOWN

        url = f"{self.base_url.rstrip('/')}/v1/classify/sentiment"
        try:
            response = requests.post(url, json={"text": text}, timeout=self.timeout_seconds)
            response.raise_for_status()
            result = SentimentResult.model_validate(response.json())
            logger.debug("classify_sentiment: text=%r -> %s", text, result)
            return result
        except requests.exceptions.RequestException as e:
            logger.warning("classify_sentiment: classifier unreachable (%s): %s", url, e)
            return SENTIMENT_UNKNOWN
        except ValueError as e:  # bad JSON, or a response shape that fails SentimentResult validation
            logger.warning("classify_sentiment: unexpected response from classifier: %s", e)
            return SENTIMENT_UNKNOWN
