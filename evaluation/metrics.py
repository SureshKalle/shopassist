# evaluation/metrics.py
"""
Two metric families, both built only from what's already in this project's
requirements.txt plus scikit-learn/numpy (see the note in requirements-eval.txt):

1. compute_routing_metrics() - standard classification F1/precision/recall
   (sklearn) over (expected_agent, actual agent_invoked) pairs. Scores
   `LLMInferenceService.call_router()` indirectly, through whatever the
   orchestrator actually did (including its own confidence-based fallback
   logic in orchestrator.py step 4) - i.e. this is an end-to-end routing
   score, not a router-in-isolation score.

2. RAGAS-style metrics for RAG-backed answers (GeneralPurposeAgent /
   MockRAGService): context_precision, context_recall, faithfulness,
   answer_relevancy - the same four metrics the `ragas` package reports,
   reimplemented here as lightweight heuristics rather than pulling in
   `ragas` itself.

   Why not the real `ragas` library: it expects an OpenAI-shaped
   chat/embeddings client (or a LangChain wrapper) to do LLM-judged
   faithfulness/relevancy scoring. This project's LLMInferenceService is a
   custom hybrid local(Ollama)/cloud(Gemini) client with its own request/
   response Pydantic models (common/models.py) - wiring that through
   LangChain's LLM interface is real integration work, not a "simple"
   addition. The heuristics below use the same signal RAGAS does
   (embedding similarity for relevancy, keyword/term overlap as a stand-in
   for LLM-judged faithfulness/precision/recall) but call
   `LLMInferenceService.call_embeddings()` directly - zero new service
   dependencies. If/when real LLM-judged scoring is wanted, swap
   `faithfulness()`/`context_precision()`/`context_recall()` for actual
   `ragas.metrics` calls behind the same function signatures used here;
   `answer_relevancy()` already does the embedding-similarity part RAGAS
   itself uses.

Every function below takes plain Python types (strings/lists/dicts) - no
dependency on services/agents/orchestrator types - so this module can be
unit-tested without spinning up the full stack.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    if va.size == 0 or vb.size == 0 or not np.any(va) or not np.any(vb):
        return 0.0
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-8
    return float(np.dot(va, vb) / denom)


# ---------------------------------------------------------------------------
# 1. Intent routing - F1 / precision / recall
# ---------------------------------------------------------------------------
def compute_routing_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """F1 (macro + weighted), per-class precision/recall/F1, and a full
    sklearn classification_report string, over agent-routing labels.

    macro-F1 treats every agent class equally regardless of how many eval
    cases target it - the more informative number when your dataset is
    small and hand-labeled (this project's ROUTING_CASES), since a class
    with only 2-3 cases shouldn't be swamped by a class with 10.
    """
    if not y_true:
        return {"error": "empty dataset - nothing to score"}

    labels = sorted(set(y_true) | set(y_pred))
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {
        label: {
            "precision": round(float(precision[i]), 3),
            "recall": round(float(recall[i]), 3),
            "f1": round(float(f1_per_class[i]), 3),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    return {
        "n_cases": len(y_true),
        "accuracy": round(accuracy, 3),
        "f1_macro": round(float(f1_macro), 3),
        "f1_weighted": round(float(f1_weighted), 3),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }


# ---------------------------------------------------------------------------
# 2. RAGAS-style RAG metrics
# ---------------------------------------------------------------------------
def context_precision(contexts: List[str], reference_keywords: List[str]) -> Optional[float]:
    """Of the chunks actually retrieved, what fraction are relevant?

    "Relevant" here = contains at least one reference keyword. Mirrors what
    RAGAS's context_precision measures (signal vs. noise in what got
    retrieved), via keyword overlap instead of an LLM relevance judgment.
    """
    if not contexts:
        return 0.0
    if not reference_keywords:
        return None
    hits = sum(1 for c in contexts if any(k.lower() in c.lower() for k in reference_keywords))
    return round(hits / len(contexts), 3)


def context_recall(contexts: List[str], reference_keywords: List[str]) -> Optional[float]:
    """Of the reference keywords a correct answer should touch on, how many
    appear *somewhere* in the retrieved contexts? Mirrors RAGAS's
    context_recall (did retrieval bring back everything needed).
    """
    if not reference_keywords:
        return None
    joined = " ".join(contexts).lower()
    hits = sum(1 for k in reference_keywords if k.lower() in joined)
    return round(hits / len(reference_keywords), 3)


def faithfulness(answer: str, contexts: List[str]) -> Optional[float]:
    """Of the content words in the answer, what fraction also appear
    somewhere in the retrieved contexts? A proxy for RAGAS's faithfulness
    (does the answer avoid claims unsupported by the retrieved context) -
    real RAGAS uses an LLM to decompose the answer into claims and checks
    each one; this is the cheap token-overlap version of the same idea, and
    will under/over-score paraphrased-but-faithful answers, so treat it as
    directional, not exact.
    """
    if not contexts:
        return 0.0
    answer_words = set(_tokenize(answer))
    # Drop very short/common tokens so the score isn't dominated by "the/a/is".
    answer_words = {w for w in answer_words if len(w) > 3}
    if not answer_words:
        return None
    context_words = set(_tokenize(" ".join(contexts)))
    supported = answer_words & context_words
    return round(len(supported) / len(answer_words), 3)


def answer_relevancy(
    llm_inference_client,
    question: str,
    answer: str,
) -> Optional[float]:
    """Cosine similarity between the question's and answer's embeddings,
    via this project's own `LLMInferenceService.call_embeddings()` - the
    same embedding-similarity signal RAGAS's answer_relevancy metric uses
    (RAGAS additionally generates synthetic questions from the answer and
    averages similarity across those; this is the single-pair simplified
    version).

    Note: `call_embeddings()` is a deterministic 16-dim stub today (see
    services/llm_inference.py), not a real embedding model - this score is
    only meaningful once that's swapped for a real embedding call/model.
    """
    try:
        q_emb = llm_inference_client.call_embeddings(question)
        a_emb = llm_inference_client.call_embeddings(answer)
    except Exception:
        logger.warning("answer_relevancy: call_embeddings failed", exc_info=True)
        return None
    return round(_cosine_similarity(q_emb, a_emb), 3)


def compute_ragas_style_metrics(
    cases: List[Dict[str, Any]],
    llm_inference_client,
) -> Dict[str, Any]:
    """Run the four metrics above over a list of already-executed cases.

    Each case dict must have: question, answer, contexts (List[str]),
    reference_keywords (List[str]). Returns per-case rows plus dataset-level
    averages (None values excluded from the average, not treated as 0).
    """
    rows = []
    for c in cases:
        row = {
            "question": c["question"],
            "answer": c["answer"],
            "n_contexts": len(c.get("contexts") or []),
            "context_precision": context_precision(c.get("contexts") or [], c.get("reference_keywords") or []),
            "context_recall": context_recall(c.get("contexts") or [], c.get("reference_keywords") or []),
            "faithfulness": faithfulness(c["answer"], c.get("contexts") or []),
            "answer_relevancy": answer_relevancy(llm_inference_client, c["question"], c["answer"]),
        }
        rows.append(row)

    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    aggregates = {
        "n_cases": len(rows),
        "avg_context_precision": _avg("context_precision"),
        "avg_context_recall": _avg("context_recall"),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_answer_relevancy": _avg("answer_relevancy"),
    }
    return {"rows": rows, "aggregates": aggregates}
