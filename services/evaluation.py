# services/evaluation.py
"""
EvaluationService — implements the platform's "9. Evaluation Framework".

Dimensions covered (target values pulled from common/config.py, sourced from
the project's evaluation framework spec):

  ┌───────────────────────┬──────────────────────────┬──────────────┬──────────┐
  │ Dimension              │ Metric                   │ Tool         │ Target   │
  ├───────────────────────┼──────────────────────────┼──────────────┼──────────┤
  │ Retrieval Quality      │ Context Recall/Precision │ RAGAS        │ ≥ 0.75   │
  │ Response Faithfulness  │ Faithfulness Score       │ RAGAS        │ ≥ 0.80   │
  │ Response Relevance     │ Answer Relevance         │ RAGAS        │ ≥ 0.75   │
  │ Semantic Accuracy      │ BERTScore F1             │ BERTScore    │ ≥ 0.75   │
  │ Response Speed         │ Avg Latency / Turn       │ Custom Timer │ < 5s     │
  │ Hallucination Control  │ % Ungrounded Claims      │ Manual+RAGAS │ < 15%    │
  │ Workflow Completion    │ % Tasks Fully Resolved   │ Test Harness │ ≥ 85%    │
  └───────────────────────┴──────────────────────────┴──────────────┴──────────┘

Design notes:
  - RAGAS metrics (Context Recall/Precision, Faithfulness, Answer Relevance)
    are imported lazily inside methods, not at module load time.

  - **RAGAS/langchain-community conflict — RESOLVED.** RAGAS (all versions
    checked: 0.2.x/0.3.x/0.4.x) unconditionally imports
    `langchain_community.chat_models.vertexai` at package-load time. That
    submodule was removed from `langchain-community` once it moved to the
    separate `langchain-google-vertexai` package — so any `langchain-community
    >= 0.4` (which our LangGraph-based agents require — see
    `common/llm_factory.py`) breaks RAGAS's import, and downgrading
    `langchain-community` to keep that submodule would in turn break our
    agents (they need `langchain-core>=1.4`, which conflicts with
    `langchain-community<0.4`'s own pin of `langchain-core<1.0`).
    Two real environments can't both be satisfied by one pip install — so
    instead of environment-splitting, `_apply_ragas_compat_shim()` below
    registers a lightweight stub module in `sys.modules` for
    `langchain_community.chat_models.vertexai` *before* RAGAS imports it.
    We never use Vertex AI in this project, so the stub is functionally
    inert — it exists purely to satisfy RAGAS's import statement. Verified
    working end-to-end (imports + `ragas.evaluate()` construction) against
    `langchain-community==0.4.2` / `ragas==0.4.3`.
  - RAGAS defaults to OpenAI for its internal judge LLM and embeddings.
    Since this project uses Groq/Ollama/Anthropic (not OpenAI — see
    `common/llm_factory.py`), every RAGAS call below explicitly wraps our
    configured chat model and embeddings via `LangchainLLMWrapper` /
    `LangchainEmbeddingsWrapper` and passes them into `ragas.evaluate()`.
    No `OPENAI_API_KEY` is required anywhere in this file.
  - `latency_timer()` is a plain context manager — no external dependency.
  - `hallucination_rate()` uses RAGAS Faithfulness under the hood (an
    "ungrounded claim" = a claim in the response not supported by the
    retrieved context), falling back to a manual keyword-overlap heuristic
    if RAGAS is unavailable (clearly flagged as lower-confidence in the result).
  - `workflow_completion_rate()` is a simple test-harness style helper: pass
    it a list of task statuses from a batch of orchestrator runs and it
    reports the % fully resolved (status == "success").
"""

import logging
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from common.config import (
    EVAL_ANSWER_RELEVANCE_TARGET,
    EVAL_BERTSCORE_F1_TARGET,
    EVAL_CONTEXT_PRECISION_TARGET,
    EVAL_CONTEXT_RECALL_TARGET,
    EVAL_FAITHFULNESS_TARGET,
    EVAL_MAX_HALLUCINATION_RATE,
    EVAL_MAX_LATENCY_SECONDS,
    EVAL_MIN_WORKFLOW_COMPLETION,
)
from common.llm_factory import build_chat_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAGAS compatibility shim — see module docstring for the full explanation.
# ---------------------------------------------------------------------------
_ragas_shim_applied = False


def _apply_ragas_compat_shim() -> None:
    """
    Register a stub for `langchain_community.chat_models.vertexai` so RAGAS's
    top-level import of `ChatVertexAI` succeeds, without requiring a
    `langchain-community` version that conflicts with our LangGraph agents.

    Idempotent — safe to call from every RAGAS-using method.
    """
    global _ragas_shim_applied
    if _ragas_shim_applied:
        return

    module_name = "langchain_community.chat_models.vertexai"
    if module_name not in sys.modules:
        stub = types.ModuleType(module_name)

        class ChatVertexAI:  # noqa: N801 — matches the real class name RAGAS imports
            """
            Inert stand-in. This project does not use Google Vertex AI;
            this class exists solely so RAGAS's `from
            langchain_community.chat_models.vertexai import ChatVertexAI`
            succeeds on import. If ever instantiated (it shouldn't be —
            we always pass our own Groq/Ollama/Anthropic LLM to RAGAS
            explicitly), it raises immediately rather than pretending to work.
            """

            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "ChatVertexAI is a compatibility stub and is not "
                    "actually implemented — this project does not use "
                    "Google Vertex AI. Pass an explicit `llm=` to RAGAS "
                    "calls instead (already done in services/evaluation.py)."
                )

        setattr(stub, "ChatVertexAI", ChatVertexAI)
        sys.modules[module_name] = stub
        logger.debug("[Evaluation] Applied RAGAS/langchain-community compatibility shim.")

    _ragas_shim_applied = True


def _build_ragas_llm_and_embeddings():
    """
    Wrap this project's configured LLM (Groq/Ollama/Anthropic) and embeddings
    model for use as RAGAS's internal judge — avoids any dependency on
    OpenAI, which RAGAS defaults to otherwise.

    Raises RuntimeError if no LLM is configured (no API key / server reachable),
    since RAGAS's LLM-based metrics cannot run without a judge model.
    """
    _apply_ragas_compat_shim()

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from services.rag import get_embeddings_model

    chat_model = build_chat_model(max_tokens=1024, temperature=0.0)
    if chat_model is None:
        raise RuntimeError(
            "No LLM available for RAGAS's judge model (check LLM_PROVIDER / "
            "API key / Ollama server in .env)."
        )

    return LangchainLLMWrapper(chat_model), LangchainEmbeddingsWrapper(get_embeddings_model())


def _import_ragas_metrics(*names: str) -> tuple:
    """
    Import module-level RAGAS metric objects (e.g. `context_recall`,
    `faithfulness`) by name, suppressing their known DeprecationWarning.

    These legacy metric objects are deprecated in RAGAS in favour of
    `ragas.metrics.collections` (a newer class-based, async-first API).
    We deliberately stay on the legacy objects: they're stable, synchronous
    (matching this codebase's sync services), and fully supported through
    RAGAS's 1.0 deprecation window — revisit this helper if/when RAGAS
    actually removes them.
    """
    import warnings

    import ragas.metrics as ragas_metrics

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return tuple(getattr(ragas_metrics, name) for name in names)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class MetricResult:
    """A single evaluated metric with its target and pass/fail status."""
    name: str
    value: Optional[float]
    target: float
    comparator: str  # ">=" or "<"
    passed: Optional[bool] = None
    note: str = ""

    def __post_init__(self):
        if self.value is None:
            self.passed = None
            return
        self.passed = self.value >= self.target if self.comparator == ">=" else self.value < self.target


@dataclass
class EvaluationReport:
    """Aggregated results across all evaluation dimensions for one test run."""
    metrics: list[MetricResult] = field(default_factory=list)

    def add(self, result: MetricResult) -> None:
        self.metrics.append(result)

    def summary(self) -> str:
        lines = ["Evaluation Report", "=" * 60]
        for m in self.metrics:
            status = "N/A" if m.passed is None else ("PASS" if m.passed else "FAIL")
            value_str = "N/A" if m.value is None else f"{m.value:.3f}"
            lines.append(
                f"  [{status:4}] {m.name:<28} value={value_str:<8} "
                f"target={m.comparator}{m.target}  {m.note}"
            )
        overall = all(m.passed for m in self.metrics if m.passed is not None)
        lines.append("=" * 60)
        lines.append(f"Overall: {'PASS' if overall else 'FAIL / INCOMPLETE'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Latency timing (Response Speed dimension)
# ---------------------------------------------------------------------------
@contextmanager
def latency_timer():
    """
    Context manager that measures wall-clock latency of a block of code.

    Usage:
        with latency_timer() as timer:
            response = orchestrator.handle_customer_query(query)
        print(timer.elapsed_seconds)
    """
    class _Timer:
        elapsed_seconds: float = 0.0

    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed_seconds = time.perf_counter() - start


class EvaluationService:
    """
    Central evaluation harness. Instantiate once and call the dimension-specific
    methods after running batches of orchestrator interactions in a test suite.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval Quality — Context Recall / Context Precision (RAGAS)
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_retrieval_quality(
        self,
        questions: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> tuple[MetricResult, MetricResult]:
        """
        Compute Context Recall and Context Precision via RAGAS.

        Args:
            questions:     list of user queries
            contexts:      list of retrieved-context lists (one list per question)
            ground_truths: list of reference/gold answers (one per question)
        """
        try:
            recall_score, precision_score = self._run_ragas_retrieval(
                questions, contexts, ground_truths
            )
        except RuntimeError as exc:
            logger.warning("[Evaluation] Retrieval quality unavailable: %s", exc)
            return (
                MetricResult("Context Recall", None, EVAL_CONTEXT_RECALL_TARGET, ">=", note=str(exc)),
                MetricResult("Context Precision", None, EVAL_CONTEXT_PRECISION_TARGET, ">=", note=str(exc)),
            )

        return (
            MetricResult("Context Recall", recall_score, EVAL_CONTEXT_RECALL_TARGET, ">="),
            MetricResult("Context Precision", precision_score, EVAL_CONTEXT_PRECISION_TARGET, ">="),
        )

    @staticmethod
    def _run_ragas_retrieval(
        questions: list[str], contexts: list[list[str]], ground_truths: list[str]
    ) -> tuple[float, float]:
        _apply_ragas_compat_shim()
        try:
            from datasets import Dataset
            from ragas import evaluate

            context_recall, context_precision = _import_ragas_metrics(
                "context_recall", "context_precision"
            )
        except ImportError as exc:
            raise RuntimeError(f"RAGAS import failed even after compat shim: {exc}") from exc

        llm, embeddings = _build_ragas_llm_and_embeddings()
        dataset = Dataset.from_dict(
            {"question": questions, "contexts": contexts, "ground_truth": ground_truths}
        )
        result = evaluate(  # type: ignore[arg-type]
            dataset, metrics=[context_recall, context_precision], llm=llm, embeddings=embeddings
        )
        return float(result["context_recall"]), float(result["context_precision"])

    # ──────────────────────────────────────────────────────────────────────────
    # Response Faithfulness (RAGAS)
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_faithfulness(
        self, questions: list[str], answers: list[str], contexts: list[list[str]]
    ) -> MetricResult:
        """Compute RAGAS Faithfulness: how well the answer is grounded in retrieved context."""
        try:
            score = self._run_ragas_faithfulness(questions, answers, contexts)
        except RuntimeError as exc:
            logger.warning("[Evaluation] Faithfulness unavailable: %s", exc)
            return MetricResult("Faithfulness", None, EVAL_FAITHFULNESS_TARGET, ">=", note=str(exc))
        return MetricResult("Faithfulness", score, EVAL_FAITHFULNESS_TARGET, ">=")

    @staticmethod
    def _run_ragas_faithfulness(
        questions: list[str], answers: list[str], contexts: list[list[str]]
    ) -> float:
        _apply_ragas_compat_shim()
        try:
            from datasets import Dataset
            from ragas import evaluate

            (faithfulness,) = _import_ragas_metrics("faithfulness")
        except ImportError as exc:
            raise RuntimeError(f"RAGAS import failed even after compat shim: {exc}") from exc

        llm, embeddings = _build_ragas_llm_and_embeddings()
        dataset = Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts})
        result = evaluate(dataset, metrics=[faithfulness], llm=llm, embeddings=embeddings)  # type: ignore[arg-type]
        return float(result["faithfulness"])

    # ──────────────────────────────────────────────────────────────────────────
    # Response Relevance (RAGAS)
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_answer_relevance(
        self, questions: list[str], answers: list[str], contexts: list[list[str]]
    ) -> MetricResult:
        """Compute RAGAS Answer Relevance: how well the answer addresses the question."""
        try:
            score = self._run_ragas_relevance(questions, answers, contexts)
        except RuntimeError as exc:
            logger.warning("[Evaluation] Answer relevance unavailable: %s", exc)
            return MetricResult("Answer Relevance", None, EVAL_ANSWER_RELEVANCE_TARGET, ">=", note=str(exc))
        return MetricResult("Answer Relevance", score, EVAL_ANSWER_RELEVANCE_TARGET, ">=")

    @staticmethod
    def _run_ragas_relevance(
        questions: list[str], answers: list[str], contexts: list[list[str]]
    ) -> float:
        _apply_ragas_compat_shim()
        try:
            from datasets import Dataset
            from ragas import evaluate

            (answer_relevancy,) = _import_ragas_metrics("answer_relevancy")
        except ImportError as exc:
            raise RuntimeError(f"RAGAS import failed even after compat shim: {exc}") from exc

        llm, embeddings = _build_ragas_llm_and_embeddings()
        dataset = Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts})
        result = evaluate(dataset, metrics=[answer_relevancy], llm=llm, embeddings=embeddings)  # type: ignore[arg-type]
        return float(result["answer_relevancy"])

    # ──────────────────────────────────────────────────────────────────────────
    # Semantic Accuracy — BERTScore F1
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_semantic_accuracy(
        self, candidates: list[str], references: list[str], lang: str = "en"
    ) -> MetricResult:
        """
        Compute BERTScore F1 between generated responses (candidates) and
        reference/gold responses. No RAGAS dependency — works standalone.

        Requires downloading a scoring model (default: roberta-large) from
        HuggingFace Hub on first use. If offline or the model can't be
        fetched, returns a metric with value=None rather than crashing.
        """
        try:
            from typing import cast

            import torch
            from bert_score import score as bert_score

            _, _, f1 = bert_score(candidates, references, lang=lang, verbose=False)
            f1 = cast(torch.Tensor, f1)
        except ImportError as exc:
            note = f"bert-score not installed ({exc}). Run: pip install bert-score"
            logger.warning("[Evaluation] %s", note)
            return MetricResult("BERTScore F1", None, EVAL_BERTSCORE_F1_TARGET, ">=", note=note)
        except Exception as exc:
            note = f"BERTScore model could not be loaded (likely offline): {exc}"
            logger.warning("[Evaluation] %s", note)
            return MetricResult("BERTScore F1", None, EVAL_BERTSCORE_F1_TARGET, ">=", note=note)

        avg_f1 = float(f1.mean())
        return MetricResult("BERTScore F1", avg_f1, EVAL_BERTSCORE_F1_TARGET, ">=")

    # ──────────────────────────────────────────────────────────────────────────
    # Response Speed — Average Latency per Turn
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_response_speed(self, latencies_seconds: list[float]) -> MetricResult:
        """
        Aggregate a list of per-turn latencies (collect these using
        `latency_timer()` around each `orchestrator.handle_customer_query()` call).
        """
        if not latencies_seconds:
            return MetricResult(
                "Avg Latency/Turn", None, EVAL_MAX_LATENCY_SECONDS, "<", note="No latency samples provided."
            )
        avg_latency = sum(latencies_seconds) / len(latencies_seconds)
        return MetricResult("Avg Latency/Turn", avg_latency, EVAL_MAX_LATENCY_SECONDS, "<")

    # ──────────────────────────────────────────────────────────────────────────
    # Hallucination Control — % Responses with Ungrounded Claims
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_hallucination_rate(
        self, questions: list[str], answers: list[str], contexts: list[list[str]]
    ) -> MetricResult:
        """
        Estimate the percentage of responses containing ungrounded claims.

        Primary method: derive from RAGAS Faithfulness (1 - faithfulness ≈
        proportion of ungrounded claims across the batch).
        Fallback (if RAGAS unavailable): a manual keyword-overlap heuristic
        per response — flagged as lower-confidence via the `note` field, per
        the framework's "Manual + RAGAS" tool column.
        """
        try:
            faithfulness_score = self._run_ragas_faithfulness(questions, answers, contexts)
            hallucination_rate = 1.0 - faithfulness_score
            return MetricResult(
                "Hallucination Rate", hallucination_rate, EVAL_MAX_HALLUCINATION_RATE, "<",
                note="Derived from RAGAS Faithfulness (1 - faithfulness).",
            )
        except RuntimeError as exc:
            logger.warning(
                "[Evaluation] RAGAS unavailable for hallucination check (%s) — "
                "using manual keyword-overlap heuristic (lower confidence).", exc,
            )
            rate = self._manual_hallucination_heuristic(answers, contexts)
            return MetricResult(
                "Hallucination Rate", rate, EVAL_MAX_HALLUCINATION_RATE, "<",
                note="Manual heuristic (RAGAS unavailable) — treat as indicative, not authoritative.",
            )

    @staticmethod
    def _manual_hallucination_heuristic(answers: list[str], contexts: list[list[str]]) -> float:
        """
        Very rough fallback: flag an answer as "possibly ungrounded" if fewer
        than 20% of its content words appear anywhere in its retrieved context.
        Intended only as a stopgap when RAGAS is unavailable.
        """
        flagged = 0
        for answer, ctx_list in zip(answers, contexts):
            context_text = " ".join(ctx_list).lower()
            answer_words = {w for w in re_split_words(answer.lower()) if len(w) > 3}
            if not answer_words:
                continue
            overlap = sum(1 for w in answer_words if w in context_text)
            coverage = overlap / len(answer_words)
            if coverage < 0.20:
                flagged += 1
        return flagged / len(answers) if answers else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Workflow Completion — % Tasks Fully Resolved by Agents
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_workflow_completion(self, statuses: list[str]) -> MetricResult:
        """
        Compute the % of orchestrator runs that fully resolved without
        escalation or failure.

        Args:
            statuses: list of `StructuredAgentResult.status` values collected
                      from a batch of test-harness runs
                      (e.g. ["success", "success", "escalation", ...]).
        """
        if not statuses:
            return MetricResult(
                "Workflow Completion", None, EVAL_MIN_WORKFLOW_COMPLETION, ">=",
                note="No task statuses provided.",
            )
        resolved = sum(1 for s in statuses if s == "success")
        rate = resolved / len(statuses)
        return MetricResult("Workflow Completion", rate, EVAL_MIN_WORKFLOW_COMPLETION, ">=")

    # ──────────────────────────────────────────────────────────────────────────
    # Full report
    # ──────────────────────────────────────────────────────────────────────────

    def build_report(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
        latencies_seconds: list[float],
        task_statuses: list[str],
    ) -> EvaluationReport:
        """
        Run every dimension of the evaluation framework in one call and
        return a consolidated EvaluationReport.
        """
        report = EvaluationReport()

        recall, precision = self.evaluate_retrieval_quality(questions, contexts, ground_truths)
        report.add(recall)
        report.add(precision)

        report.add(self.evaluate_faithfulness(questions, answers, contexts))
        report.add(self.evaluate_answer_relevance(questions, answers, contexts))
        report.add(self.evaluate_semantic_accuracy(answers, ground_truths))
        report.add(self.evaluate_response_speed(latencies_seconds))
        report.add(self.evaluate_hallucination_rate(questions, answers, contexts))
        report.add(self.evaluate_workflow_completion(task_statuses))

        return report


def re_split_words(text: str) -> list[str]:
    """Small helper kept local to avoid importing `re` at module top just for this."""
    import re

    return re.findall(r"\b\w+\b", text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    evaluator = EvaluationService()

    # ── Response Speed demo (no external deps) ──────────────────────────────
    with latency_timer() as t:
        time.sleep(0.05)
    print(f"Sample latency: {t.elapsed_seconds:.3f}s")

    speed_result = evaluator.evaluate_response_speed([0.5, 1.2, 4.8, 0.9])
    print(speed_result)

    # ── Workflow completion demo ─────────────────────────────────────────────
    completion_result = evaluator.evaluate_workflow_completion(
        ["success", "success", "escalation", "success", "failure"]
    )
    print(completion_result)

    # ── Semantic accuracy demo (BERTScore — works standalone) ───────────────
    semantic_result = evaluator.evaluate_semantic_accuracy(
        candidates=["Your order has shipped and will arrive soon."],
        references=["Your order shipped and is on its way to you."],
    )
    print(semantic_result)

    # ── Full report (RAGAS-backed dims will show "N/A" if RAGAS unavailable) ─
    report = evaluator.build_report(
        questions=["Where is my order?"],
        answers=["Your order 12345 has shipped and will arrive by Friday."],
        contexts=[["Order 12345 status: shipped. Estimated delivery: Friday."]],
        ground_truths=["Order 12345 shipped, arriving Friday."],
        latencies_seconds=[1.2],
        task_statuses=["success"],
    )
    print("\n" + report.summary())
