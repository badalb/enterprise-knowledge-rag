import argparse

from datetime import datetime, timezone

from pathlib import Path

from evaluation.dataset_loader import (
    load_dataset,
    DEFAULT_DATASET_PATH
)

from evaluation.generation_evaluator import (
    GenerationEvaluator
)

from evaluation.models import (
    GenerationEvaluationReport
)


REPORT_DIRECTORY = (
    Path(__file__).resolve().parent / "reports"
)


def format_score(value):

    return (
        f"{value:.4f}"
        if value is not None
        else "N/A"
    )


def format_number(value):

    return (
        f"{value:,.2f}"
        if value is not None
        else "N/A"
    )


def print_report(
    report: GenerationEvaluationReport
):

    d = report.summary.deterministic

    s = report.summary.semantic

    print("\n" + "=" * 70)

    print("GENERATION EVALUATION REPORT")

    print("=" * 70)

    print(f"Embedding model: {report.embedding_model}")

    print(f"Generation model: {report.generation_model}")

    print(f"Judge model: {report.judge_model}")

    print(f"Indexed chunks: {report.indexed_chunks}")

    print(f"Top-K: {report.top_k}")

    print(
        f"Similarity threshold: "
        f"{report.min_similarity_score}"
    )

    print(f"Total cases: {d.total_cases}")

    print(f"Answerable cases: {d.answerable_cases}")

    print(f"Unanswerable cases: {d.unanswerable_cases}")

    print("\nDETERMINISTIC METRICS")

    deterministic_metrics = {
        "Abstention accuracy":
            d.abstention_accuracy,

        "Answerable response rate":
            d.answerable_response_rate,

        "False abstention rate":
            d.false_abstention_rate,

        "Unanswerable rejection rate":
            d.unanswerable_rejection_rate,

        "Citation validity rate":
            d.citation_validity_rate,

        "Citation coverage rate":
            d.citation_coverage_rate,

        "Generation invocation rate":
            d.generation_invocation_rate,

        "Model abstention rate":
            d.model_abstention_rate,

        "Guardrail intervention rate":
            d.guardrail_intervention_rate
    }

    for name, value in deterministic_metrics.items():

        print(f"{name}: {format_score(value)}")

    print("\nABSTENTION CONFUSION MATRIX")

    print(f"True positives: {d.true_positives}")

    print(f"True negatives: {d.true_negatives}")

    print(f"False positives: {d.false_positives}")

    print(f"False negatives: {d.false_negatives}")

    print("\nABSTENTION REASONS")

    for reason, count in sorted(
        d.abstention_reason_counts.items()
    ):

        print(f"{reason}: {count}")

    print("\nSEMANTIC METRICS")

    print(f"Judged cases: {s.judged_cases}")

    print(f"Skipped cases: {s.skipped_cases}")

    semantic_metrics = {
        "Conditional correctness":
            s.mean_correctness,

        "Mean faithfulness":
            s.mean_faithfulness,

        "Mean answer relevance":
            s.mean_answer_relevance,

        "Mean citation support":
            s.mean_citation_support,

        "Conditional fact coverage":
            s.mean_fact_coverage,

        "End-to-end correctness":
            s.end_to_end_correctness,

        "End-to-end fact coverage":
            s.end_to_end_fact_coverage,

        "Micro fact coverage":
            s.micro_fact_coverage
    }

    for name, value in semantic_metrics.items():

        print(f"{name}: {format_score(value)}")

    print(
        f"Expected facts covered: "
        f"{s.total_covered_facts}/"
        f"{s.total_expected_facts}"
    )

    print(
        f"Unsupported claims: "
        f"{s.unsupported_claim_count}"
    )

    print(
        f"Contradictory claims: "
        f"{s.contradictory_claim_count}"
    )

    print(
        f"Citation issues: "
        f"{s.citation_issue_count}"
    )

    print("\nAPPLICATION PERFORMANCE")

    performance_metrics = {
        "Mean retrieved chunks":
            d.mean_retrieved_chunks,

        "Total generation input tokens":
            d.total_input_tokens,

        "Total generation output tokens":
            d.total_output_tokens,

        "Mean generation input tokens":
            d.mean_input_tokens,

        "Mean generation output tokens":
            d.mean_output_tokens,

        "Mean retrieval latency (ms)":
            d.mean_retrieval_latency_ms,

        "Mean generation latency (ms)":
            d.mean_generation_latency_ms,

        "Mean total latency (ms)":
            d.mean_total_latency_ms,

        "P95 total latency (ms)":
            d.p95_total_latency_ms
    }

    for name, value in performance_metrics.items():

        print(f"{name}: {format_number(value)}")

    print("\nEVALUATION OVERHEAD")

    print(
        f"Total judge input tokens: "
        f"{format_number(s.total_judge_input_tokens)}"
    )

    print(
        f"Total judge output tokens: "
        f"{format_number(s.total_judge_output_tokens)}"
    )

    print(
        f"Mean judge latency (ms): "
        f"{format_number(s.mean_judge_latency_ms)}"
    )

    print(
        f"Evaluation duration (ms): "
        f"{format_number(report.evaluation_duration_ms)}"
    )

    print("\nCATEGORY SUMMARY")

    for category, summary in sorted(
        report.category_summaries.items()
    ):

        print(f"\n{category}")

        print(
            f"  Cases: "
            f"{summary.deterministic.total_cases}"
        )

        print(
            f"  Abstention accuracy: "
            f"{format_score(summary.deterministic.abstention_accuracy)}"
        )

        print(
            f"  End-to-end correctness: "
            f"{format_score(summary.semantic.end_to_end_correctness)}"
        )

        print(
            f"  Faithfulness: "
            f"{format_score(summary.semantic.mean_faithfulness)}"
        )

        print(
            f"  End-to-end fact coverage: "
            f"{format_score(summary.semantic.end_to_end_fact_coverage)}"
        )

    print("\nFAILED OR IMPERFECT CASES")

    failures = []

    for result in report.results:

        semantic = result.semantic

        deterministic = result.deterministic

        semantic_failure = (
            semantic is not None
            and (
                semantic.correctness < 1.0

                or semantic.faithfulness < 1.0

                or semantic.answer_relevance < 1.0

                or semantic.citation_support < 1.0

                or (
                    semantic.fact_coverage is not None
                    and semantic.fact_coverage < 1.0
                )

                or bool(semantic.unsupported_claims)

                or bool(semantic.contradictory_claims)

                or bool(semantic.citation_issues)
            )
        )

        if (
            not deterministic.abstention_correct
            or deterministic.guardrail_intervened
            or semantic_failure
        ):

            failures.append(result)

    if not failures:

        print("No failed or imperfect cases.")

    for result in failures:

        print(f"\n{result.case.id}")

        print(f"Question: {result.case.question}")

        print(f"Answer: {result.trace.response.answer}")

        print(
            f"Abstention reason: "
            f"{result.trace.abstention_reason}"
        )

        if result.semantic is not None:

            print(
                f"Missing facts: "
                f"{result.semantic.missing_facts}"
            )

            print(
                f"Unsupported claims: "
                f"{result.semantic.unsupported_claims}"
            )

            print(
                f"Contradictory claims: "
                f"{result.semantic.contradictory_claims}"
            )

            print(
                f"Citation issues: "
                f"{result.semantic.citation_issues}"
            )

            print(
                f"Judge explanation: "
                f"{result.semantic.reasoning}"
            )


def save_report(
    report: GenerationEvaluationReport
) -> Path:

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    filename = (
        f"generation_k{report.top_k}"
        f"_threshold{report.min_similarity_score}"
        f"_{timestamp}.json"
    )

    path = REPORT_DIRECTORY / filename

    path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8"
    )

    return path


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate RAG generation quality"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH
    )

    args = parser.parse_args()

    if args.top_k is not None:

        if not 1 <= args.top_k <= 20:

            parser.error(
                "--top-k must be between 1 and 20"
            )

    if args.threshold is not None:

        if not -1.0 <= args.threshold <= 1.0:

            parser.error(
                "--threshold must be between -1 and 1"
            )

    cases = load_dataset(args.dataset)

    evaluator = GenerationEvaluator()

    report = evaluator.evaluate(
        cases=cases,

        top_k=args.top_k,

        min_similarity_score=args.threshold
    )

    path = save_report(report)

    print_report(report)

    print(f"\nReport saved: {path}")


if __name__ == "__main__":

    main()