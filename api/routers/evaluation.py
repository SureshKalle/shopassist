# api/routers/evaluation.py
"""
Evaluation endpoint — run the full 7-dimension Evaluation Framework
(see services/evaluation.py) against a labelled test batch.

POST /api/v1/evaluate
"""

import logging

from fastapi import APIRouter, Depends

from api.dependencies import get_evaluation_service
from api.schemas import EvaluationRequest, EvaluationResponse, MetricResultOut
from services.evaluation import EvaluationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/evaluate", tags=["evaluation"])


@router.post("", response_model=EvaluationResponse)
def run_evaluation(
    request: EvaluationRequest,
    evaluator: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationResponse:
    """
    Run the full evaluation framework against a batch of question/answer/
    context/ground-truth tuples plus optional latency and task-status samples.

    RAGAS-backed dimensions (Context Recall/Precision, Faithfulness, Answer
    Relevance) require a configured LLM (LLM_PROVIDER in .env) and will
    return `value: null` with an explanatory `note` if unavailable, rather
    than failing the whole request.
    """
    report = evaluator.build_report(
        questions=request.questions,
        answers=request.answers,
        contexts=request.contexts,
        ground_truths=request.ground_truths,
        latencies_seconds=request.latencies_seconds,
        task_statuses=request.task_statuses,
    )

    metrics_out = [
        MetricResultOut(
            name=m.name, value=m.value, target=m.target,
            comparator=m.comparator, passed=m.passed, note=m.note,
        )
        for m in report.metrics
    ]
    overall_pass = all(m.passed for m in report.metrics if m.passed is not None)

    return EvaluationResponse(metrics=metrics_out, overall_pass=overall_pass)
