from datetime import datetime, timezone
from pathlib import Path

from evaluation.dataset_loader import load_dataset

from evaluation.retrieval_experiment import (
    RetrievalExperiment
)

from evaluation.models import (
    RetrievalExperimentResult
)


REPORT_DIRECTORY = (
    Path(__file__).resolve().parent / "reports"
)


def score(value: float | None) -> str:

    return (
        f"{value:.4f}"
        if value is not None
        else "N/A"
    )


def print_result(
    result: RetrievalExperimentResult
):

    print(f"Top-K: {result.top_k}")

    print(
        f"Threshold: "
        f"{result.min_similarity_score}"
    )

    print(
        f"Precision@K: "
        f"{score(result.mean_precision_at_k)}"
    )

    print(
        f"Recall@K: "
        f"{score(result.mean_recall_at_k)}"
    )

    print(
        f"MRR: "
        f"{score(result.mean_reciprocal_rank)}"
    )

    print(
        f"Context Precision: "
        f"{score(result.mean_context_precision)}"
    )

    print(
        f"Rejection Rate: "
        f"{score(result.retrieval_rejection_rate)}"
    )

    print(
        f"Full Recall Rate: "
        f"{score(result.full_recall_rate)}"
    )

    print(
        f"Mean Retrieved Chunks: "
        f"{result.mean_retrieved_chunks:.2f}"
    )


def main():

    cases = load_dataset()

    experiment = RetrievalExperiment()

    report = experiment.run(cases)

    print("\n" + "=" * 90)
    print("RETRIEVAL PARAMETER EXPERIMENT")
    print("=" * 90)

    print(
        f"Total configurations: "
        f"{report.total_configurations}"
    )

    print(
        f"Total cases: {report.total_cases}"
    )

    print(
        f"Candidate retrieval time: "
        f"{report.candidate_retrieval_latency_ms:.2f} ms"
    )

    print("\nCONFIGURATION COMPARISON")
    print("-" * 90)

    print(
        f"{'K':>3} "
        f"{'Threshold':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'MRR':>10} "
        f"{'Context-P':>10} "
        f"{'Reject':>10} "
        f"{'Chunks':>8}"
    )

    print("-" * 90)

    for result in report.results:

        print(
            f"{result.top_k:>3} "
            f"{result.min_similarity_score:>10.2f} "
            f"{score(result.mean_precision_at_k):>10} "
            f"{score(result.mean_recall_at_k):>10} "
            f"{score(result.mean_reciprocal_rank):>10} "
            f"{score(result.mean_context_precision):>10} "
            f"{score(result.retrieval_rejection_rate):>10} "
            f"{result.mean_retrieved_chunks:>8.2f}"
        )

    print("\nBASELINE CONFIGURATION")
    print("-" * 90)

    print_result(report.baseline)

    print("\nRECOMMENDED CONFIGURATION")
    print("-" * 90)

    if report.recommended is None:

        print(
            "No configuration satisfies "
            "the recall and rejection constraints."
        )

    else:

        print_result(report.recommended)

        baseline_cp = (
            report.baseline.mean_context_precision
            or 0.0
        )

        recommended_cp = (
            report.recommended.mean_context_precision
            or 0.0
        )

        improvement = (
            recommended_cp - baseline_cp
        ) * 100

        baseline_chunks = (
            report.baseline.mean_retrieved_chunks
        )

        recommended_chunks = (
            report.recommended.mean_retrieved_chunks
        )

        reduction = (
            (baseline_chunks - recommended_chunks)
            / baseline_chunks * 100
            if baseline_chunks > 0
            else 0.0
        )

        print("\nIMPROVEMENT OVER BASELINE")
        print("-" * 90)

        print(
            f"Context precision improvement: "
            f"{improvement:.2f} percentage points"
        )

        print(
            f"Retrieved chunk reduction: "
            f"{reduction:.2f}%"
        )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    report_path = (
        REPORT_DIRECTORY
        / f"retrieval_experiment_{timestamp}.json"
    )

    report_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8"
    )

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":

    main()