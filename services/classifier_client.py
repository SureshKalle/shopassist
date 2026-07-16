# services/classifier_client.py
"""Thin client for shopassist-model's classifier service — the
encoder-family (BERT-style) sentiment and topic classification models,
served separately from Ollama's decoder-family chat models. See
shopassist-model's README ("Encoder models: sentiment & topic
classification") for why these need a separate serving path: Ollama has no
way to load an arbitrary Hugging Face AutoModelForSequenceClassification
checkpoint, only GGUF decoder checkpoints.

Plain synchronous `requests`, matching this repo's style elsewhere (the
`openai` client and SQLAlchemy engine are both sync too — no async/await
anywhere in this codebase despite FastAPI).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

CLASSIFIER_API_BASE_URL = os.getenv("CLASSIFIER_API_BASE_URL", "http://localhost:8100")


class ClassifierClient:
    """Calls shopassist-model's /v1/classify/* endpoints.

    Both methods fail soft: on any request error (classifier unreachable,
    timeout, non-2xx), they log a warning and return a neutral/"other"
    default rather than raising. Classification enriches ingestion data —
    it shouldn't be able to take the pipeline down over a transient
    classifier hiccup.
    """

    def __init__(self, base_url: str = CLASSIFIER_API_BASE_URL, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def classify_sentiment(self, text: str) -> Dict[str, Any]:
        """Returns {"label", "stars", "score", "raw_label"}."""
        try:
            resp = requests.post(
                f"{self.base_url}/v1/classify/sentiment",
                json={"text": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            print(f"  [ClassifierClient] sentiment classification failed, defaulting to neutral: {exc}")
            return {"label": "neutral", "stars": 3, "score": 0.0, "raw_label": "unavailable"}

    def classify_feedback(self, text: str, labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Returns {"label", "score", "all_scores"}."""
        try:
            resp = requests.post(
                f"{self.base_url}/v1/classify/feedback",
                json={"text": text, "labels": labels},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            print(f"  [ClassifierClient] feedback classification failed, defaulting to 'other': {exc}")
            return {"label": "other", "score": 0.0, "all_scores": {}}
