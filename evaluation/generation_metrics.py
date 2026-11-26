import math
import re

from collections import Counter
from collections.abc import Sequence

from app.models.execution_trace import RAGExecutionTrace
from app.services.rag_service import RAGService

from evaluation.models import (
    EvaluationCase,
    DeterministicEvaluationResult,
    DeterministicEvaluationSummary
)

from evaluation.metrics import mean_metric

from app.generation.citation_parser import CitationParser


GUARDRAIL_REASONS = {
    "INVALID_CITATIONS",
    "MISSING_CITATIONS"
}


def safe_divide(
    numerator: int | float,
    denominator: int | float
) -> float | None:

    if denominator == 0:

        return None

    return numerator / denominator


def percentile(
    values: Sequence[float],
    p: float
) -> float:

    if not values:

        raise ValueError("Values cannot be empty")

    if not 0 < p <= 100:

        raise ValueError(
            "Percentile must be between 0 and 100"
        )

    ordered = sorted(values)

    index = math.ceil(
        p / 100 * len(ordered)
    ) - 1

    return ordered[index]


def evaluate_deterministic(
    case: EvaluationCase,
    trace: RAGExecutionTrace
) -> DeterministicEvaluationResult:

    if case.question.strip() != trace.question.strip():

        raise ValueError(
            f"Question mismatch for {case.id}"
        )

    expected_abstained = not case.answerable

    abstention_correct = (
        expected_abstained == trace.abstained
    )

    raw_answer = trace.raw_answer or ""

    model_abstained = (
        trace.generation_attempted
        and raw_answer.strip()
        == RAGService.INSUFFICIENT_CONTEXT
    )

    guardrail_intervened = (
        trace.abstention_reason
        in GUARDRAIL_REASONS
    )

    citation_valid = None

    citation_present = None

    # Evaluate raw generated answers, not just
    # answers that passed application validation.

    if trace.generation_attempted and not model_abstained:

        citation_ids = CitationParser.extract_ids(raw_answer)

        citation_present = bool(citation_ids)

        citation_valid = (
            bool(citation_ids)
            and not trace.invalid_citations
            and all(
                1 <= citation_id <= len(
                    trace.retrieved_chunks
                )
                for citation_id in citation_ids
            )
        )

    metrics = trace.response.metrics

    return DeterministicEvaluationResult(
        case_id=case.id,

        category=case.category,

        expected_answerable=case.answerable,

        actual_abstained=trace.abstained,

        abstention_correct=abstention_correct,

        abstention_reason=trace.abstention_reason,

        generation_attempted=(
            trace.generation_attempted
        ),

        model_abstained=model_abstained,

        guardrail_intervened=guardrail_intervened,

        citation_valid=citation_valid,

        citation_present=citation_present,

        retrieved_chunks=len(
            trace.retrieved_chunks
        ),

        input_tokens=metrics.input_tokens,

        output_tokens=metrics.output_tokens,

        retrieval_latency_ms=(
            metrics.retrieval_latency_ms
        ),

        generation_latency_ms=(
            metrics.generation_latency_ms
        ),

        total_latency_ms=(
            metrics.total_latency_ms
        )
    )


def summarize_deterministic(
    results: list[DeterministicEvaluationResult]
) -> DeterministicEvaluationSummary:

    if not results:

        raise ValueError(
            "Evaluation results cannot be empty"
        )

    total = len(results)

    answerable = [
        result
        for result in results
        if result.expected_answerable
    ]

    unanswerable = [
        result
        for result in results
        if not result.expected_answerable
    ]

    generated = [
        result
        for result in results
        if result.generation_attempted
    ]

    tp = sum(
        not result.actual_abstained
        for result in answerable
    )

    fn = sum(
        result.actual_abstained
        for result in answerable
    )

    tn = sum(
        result.actual_abstained
        for result in unanswerable
    )

    fp = sum(
        not result.actual_abstained
        for result in unanswerable
    )

    valid_citations = [
        float(result.citation_valid)
        for result in results
        if result.citation_valid is not None
    ]

    present_citations = [
        float(result.citation_present)
        for result in results
        if result.citation_present is not None
    ]

    abstention_reasons = Counter(
        result.abstention_reason
        for result in results
        if result.abstention_reason is not None
    )

    input_tokens = [
        result.input_tokens
        for result in results
        if result.input_tokens is not None
    ]

    output_tokens = [
        result.output_tokens
        for result in results
        if result.output_tokens is not None
    ]

    total_latencies = [
        result.total_latency_ms
        for result in results
    ]

    return DeterministicEvaluationSummary(
        total_cases=total,

        answerable_cases=len(answerable),

        unanswerable_cases=len(unanswerable),

        true_positives=tp,

        true_negatives=tn,

        false_positives=fp,

        false_negatives=fn,

        abstention_accuracy=(tp + tn) / total,

        answerable_response_rate=safe_divide(
            tp,
            len(answerable)
        ),

        false_abstention_rate=safe_divide(
            fn,
            len(answerable)
        ),

        unanswerable_rejection_rate=safe_divide(
            tn,
            len(unanswerable)
        ),

        citation_validity_rate=mean_metric(
            valid_citations
        ),

        citation_coverage_rate=mean_metric(
            present_citations
        ),

        generation_invocation_rate=(
            len(generated) / total
        ),

        model_abstention_rate=safe_divide(
            sum(
                result.model_abstained
                for result in generated
            ),
            len(generated)
        ),

        guardrail_intervention_rate=safe_divide(
            sum(
                result.guardrail_intervened
                for result in generated
            ),
            len(generated)
        ),

        abstention_reason_counts=dict(
            abstention_reasons
        ),

        mean_retrieved_chunks=(
            sum(
                result.retrieved_chunks
                for result in results
            ) / total
        ),

        total_input_tokens=(
            sum(input_tokens)
            if len(input_tokens) == total
            else None
        ),

        total_output_tokens=(
            sum(output_tokens)
            if len(output_tokens) == total
            else None
        ),

        mean_input_tokens=(
            mean_metric(input_tokens)
            if len(input_tokens) == total
            else None
        ),

        mean_output_tokens=(
            mean_metric(output_tokens)
            if len(output_tokens) == total
            else None
        ),

        mean_retrieval_latency_ms=(
            sum(
                result.retrieval_latency_ms
                for result in results
            ) / total
        ),

        mean_generation_latency_ms=mean_metric([
            result.generation_latency_ms
            for result in generated
        ]),

        mean_total_latency_ms=(
            sum(total_latencies) / total
        ),

        p95_total_latency_ms=percentile(
            total_latencies,
            95
        )
    )