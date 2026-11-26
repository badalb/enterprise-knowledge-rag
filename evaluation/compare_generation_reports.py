import argparse
import json

from pathlib import Path
from datetime import datetime, timezone

from evaluation.models import GenerationEvaluationReport


def load_report(path: Path) -> GenerationEvaluationReport:

    return GenerationEvaluationReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def validate_comparison(baseline, optimized):

    fields = [
        "embedding_model",
        "generation_model",
        "judge_model",
        "collection_name",
        "indexed_chunks"
    ]

    for field in fields:

        if getattr(baseline, field) != getattr(optimized, field):

            raise ValueError(
                f"Incompatible reports: {field}"
            )

    baseline_cases = {
        result.case.id: result.case.model_dump()
        for result in baseline.results
    }

    optimized_cases = {
        result.case.id: result.case.model_dump()
        for result in optimized.results
    }

    if baseline_cases != optimized_cases:

        raise ValueError(
            "Reports contain different evaluation cases"
        )


def compare_metric(name, baseline, optimized, unit=""):

    delta = (
        optimized - baseline
        if baseline is not None and optimized is not None
        else None
    )

    relative_change = (
        delta / baseline * 100
        if delta is not None and baseline != 0
        else None
    )

    return {
        "metric": name,
        "baseline": baseline,
        "optimized": optimized,
        "absolute_change": delta,
        "relative_change_percent": relative_change,
        "unit": unit
    }


def compare_reports(baseline, optimized):

    validate_comparison(baseline, optimized)

    bd = baseline.summary.deterministic
    od = optimized.summary.deterministic

    bs = baseline.summary.semantic
    os = optimized.summary.semantic

    metrics = []

    deterministic_fields = [
        "abstention_accuracy",
        "answerable_response_rate",
        "false_abstention_rate",
        "unanswerable_rejection_rate",
        "citation_validity_rate",
        "citation_coverage_rate",
        "generation_invocation_rate",
        "guardrail_intervention_rate",
        "mean_retrieved_chunks",
        "total_input_tokens",
        "total_output_tokens",
        "mean_retrieval_latency_ms",
        "mean_generation_latency_ms",
        "mean_total_latency_ms",
        "p95_total_latency_ms"
    ]

    for field in deterministic_fields:

        metrics.append(
            compare_metric(
                field,
                getattr(bd, field),
                getattr(od, field)
            )
        )

    semantic_fields = [
        "mean_correctness",
        "mean_faithfulness",
        "mean_answer_relevance",
        "mean_citation_support",
        "mean_fact_coverage",
        "end_to_end_correctness",
        "end_to_end_fact_coverage",
        "micro_fact_coverage",
        "total_covered_facts",
        "unsupported_claim_count",
        "contradictory_claim_count",
        "citation_issue_count",
        "total_judge_input_tokens",
        "total_judge_output_tokens"
    ]

    for field in semantic_fields:

        metrics.append(
            compare_metric(
                field,
                getattr(bs, field),
                getattr(os, field)
            )
        )

    return metrics


def case_correctness(result):

    if not result.case.answerable:

        return None

    if result.trace.abstained:

        return 0.0

    if result.semantic is None:

        raise ValueError(
            f"Missing semantic assessment: {result.case.id}"
        )

    return result.semantic.correctness


def compare_cases(baseline, optimized):

    optimized_by_id = {
        result.case.id: result
        for result in optimized.results
    }

    comparisons = []

    for before in baseline.results:

        after = optimized_by_id[before.case.id]

        before_score = case_correctness(before)
        after_score = case_correctness(after)

        correctness_change = (
            after_score - before_score
            if before_score is not None
            and after_score is not None
            else None
        )

        comparisons.append({
            "case_id": before.case.id,

            "question": before.case.question,

            "category": before.case.category,

            "answerable": before.case.answerable,

            "baseline_chunks": len(
                before.trace.retrieved_chunks
            ),

            "optimized_chunks": len(
                after.trace.retrieved_chunks
            ),

            "baseline_abstained": before.trace.abstained,

            "optimized_abstained": after.trace.abstained,

            "baseline_abstention_reason":
                before.trace.abstention_reason,

            "optimized_abstention_reason":
                after.trace.abstention_reason,

            "baseline_correctness": before_score,

            "optimized_correctness": after_score,

            "correctness_change": correctness_change
        })

    return comparisons


def format_value(value):

    return (
        f"{value:.4f}"
        if value is not None
        else "N/A"
    )


def print_comparison(metrics, cases):

    print("\nGENERATION BENCHMARK COMPARISON")

    print("=" * 100)

    print(
        f"{'Metric':<40}"
        f"{'Baseline':>15}"
        f"{'Optimized':>15}"
        f"{'Change':>15}"
    )

    print("-" * 100)

    for metric in metrics:

        print(
            f"{metric['metric']:<40}"
            f"{format_value(metric['baseline']):>15}"
            f"{format_value(metric['optimized']):>15}"
            f"{format_value(metric['absolute_change']):>15}"
        )

    print("\nCASE-LEVEL CORRECTNESS CHANGES")

    for case in cases:

        change = case["correctness_change"]

        if change is None or change == 0:

            continue

        status = (
            "IMPROVED"
            if change > 0
            else "REGRESSED"
        )

        print(
            f"{case['case_id']} | "
            f"{status} | "
            f"{case['baseline_correctness']:.2f} -> "
            f"{case['optimized_correctness']:.2f}"
        )

        print(f"  {case['question']}")

    print("\nRECOVERED FALSE ABSTENTIONS")

    for case in cases:

        if (
            case["answerable"]
            and case["baseline_abstained"]
            and not case["optimized_abstained"]
        ):

            print(
                f"{case['case_id']} | "
                f"{case['baseline_abstention_reason']} "
                f"-> ANSWERED"
            )


def main():

    parser = argparse.ArgumentParser(
        description="Compare generation evaluation reports"
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--optimized",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    baseline = load_report(args.baseline)

    optimized = load_report(args.optimized)

    metrics = compare_reports(baseline, optimized)

    cases = compare_cases(baseline, optimized)

    print_comparison(metrics, cases)

    comparison = {
        "compared_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "baseline_report": str(args.baseline),

        "optimized_report": str(args.optimized),

        "baseline_configuration": {
            "top_k": baseline.top_k,
            "threshold": baseline.min_similarity_score
        },

        "optimized_configuration": {
            "top_k": optimized.top_k,
            "threshold": optimized.min_similarity_score
        },

        "metrics": metrics,

        "case_comparisons": cases
    }

    report_directory = (
        Path(__file__).resolve().parent / "reports"
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    path = (
        report_directory /
        f"generation_comparison_{timestamp}.json"
    )

    path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8"
    )

    print(f"\nComparison saved: {path}")


if __name__ == "__main__":

    main()