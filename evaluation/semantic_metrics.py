from evaluation.models import (
    GenerationEvaluationResult,
    SemanticEvaluationSummary
)

from evaluation.metrics import mean_metric

from evaluation.generation_metrics import safe_divide


def summarize_semantic(
    results: list[GenerationEvaluationResult]
) -> SemanticEvaluationSummary:

    if not results:

        raise ValueError(
            "Evaluation results cannot be empty"
        )

    judged = [
        result.semantic
        for result in results
        if result.semantic is not None
    ]

    answerable = [
        result
        for result in results
        if result.case.answerable
    ]

    judged_answerable = [
        result.semantic
        for result in answerable
        if result.semantic is not None
    ]

    # Penalize false abstentions with zero.
    # A missing assessment for a returned answer
    # is an evaluation failure, not an abstention.

    for result in results:

        if (
            not result.trace.abstained
            and result.semantic is None
        ):

            raise ValueError(
                f"Missing semantic assessment: "
                f"{result.case.id}"
            )

    end_to_end_correctness = mean_metric([
        result.semantic.correctness
        if result.semantic is not None
        else 0.0
        for result in answerable
    ])

    end_to_end_fact_coverage = mean_metric([
        result.semantic.fact_coverage
        if result.semantic is not None
        else 0.0
        for result in answerable
    ])

    total_expected_facts = sum(
        len(result.case.expected_facts)
        for result in answerable
    )

    total_covered_facts = sum(
        len(result.covered_facts)
        for result in judged_answerable
    )

    judge_input_tokens = [
        result.judge_input_tokens
        for result in judged
        if result.judge_input_tokens is not None
    ]

    judge_output_tokens = [
        result.judge_output_tokens
        for result in judged
        if result.judge_output_tokens is not None
    ]

    return SemanticEvaluationSummary(
        total_cases=len(results),

        judged_cases=len(judged),

        skipped_cases=len(results) - len(judged),

        judged_answerable_cases=len(
            judged_answerable
        ),

        mean_correctness=mean_metric([
            result.correctness
            for result in judged_answerable
        ]),

        mean_faithfulness=mean_metric([
            result.faithfulness
            for result in judged
        ]),

        mean_answer_relevance=mean_metric([
            result.answer_relevance
            for result in judged
        ]),

        mean_citation_support=mean_metric([
            result.citation_support
            for result in judged
        ]),

        mean_fact_coverage=mean_metric([
            result.fact_coverage
            for result in judged_answerable
        ]),

        end_to_end_correctness=(
            end_to_end_correctness
        ),

        end_to_end_fact_coverage=(
            end_to_end_fact_coverage
        ),

        total_expected_facts=(
            total_expected_facts
        ),

        total_covered_facts=(
            total_covered_facts
        ),

        micro_fact_coverage=safe_divide(
            total_covered_facts,
            total_expected_facts
        ),

        unsupported_claim_count=sum(
            len(result.unsupported_claims)
            for result in judged
        ),

        contradictory_claim_count=sum(
            len(result.contradictory_claims)
            for result in judged
        ),

        citation_issue_count=sum(
            len(result.citation_issues)
            for result in judged
        ),

        total_judge_input_tokens=(
            sum(judge_input_tokens)
            if len(judge_input_tokens) == len(judged)
            else None
        ),

        total_judge_output_tokens=(
            sum(judge_output_tokens)
            if len(judge_output_tokens) == len(judged)
            else None
        ),

        mean_judge_latency_ms=mean_metric([
            result.judge_latency_ms
            for result in judged
        ])
    )