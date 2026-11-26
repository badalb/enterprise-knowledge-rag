import argparse

from datetime import datetime, timezone
from pathlib import Path

from evaluation.dataset_loader import (
    load_dataset,
    DEFAULT_DATASET_PATH
)

from evaluation.models import (
    RetrievalEvaluationReport,
    RetrievalEvaluationSummary
)

from evaluation.retrieval_evaluator import (
    RetrievalEvaluator
)


REPORT_DIRECTORY = (
    Path(__file__).resolve().parent / "reports"
)


def format_score(value: float | None) -> str:

    return (
        f"{value:.4f}"
        if value is not None
        else "N/A"
    )


def print_summary(
    summary: RetrievalEvaluationSummary
):

    print(f"Total cases: {summary.total_cases}")

    print(
        f"Evaluated relevance cases: "
        f"{summary.evaluated_cases}"
    )

    print(
        f"No expected document cases: "
        f"{summary.no_expected_document_cases}"
    )

    print(
        f"Mean Precision@K: "
        f"{format_score(summary.mean_precision_at_k)}"
    )

    print(
        f"Mean Recall@K: "
        f"{format_score(summary.mean_recall_at_k)}"
    )

    print(
        f"MRR: "
        f"{format_score(summary.mean_reciprocal_rank)}"
    )

    print(
        f"Hit Rate: "
        f"{format_score(summary.hit_rate)}"
    )

    print(
        f"Mean Context Precision: "
        f"{format_score(summary.mean_context_precision)}"
    )

    print(
        f"Retrieval Rejection Rate: "
        f"{format_score(summary.no_relevant_document_rejection_rate)}"
    )

    print(
        f"Mean Retrieved Chunks: "
        f"{summary.mean_retrieved_chunks:.2f}"
    )

    print(
        f"Mean Latency: "
        f"{summary.mean_latency_ms:.2f} ms"
    )

    print(
        f"P95 Latency: "
        f"{summary.p95_latency_ms:.2f} ms"
    )


def print_report(
    report: RetrievalEvaluationReport
):

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 70)

    print(f"Embedding model: {report.embedding_model}")

    print(f"Indexed chunks: {report.indexed_chunks}")

    print(f"Top-K: {report.top_k}")

    print(
        f"Similarity threshold: "
        f"{report.min_similarity_score}"
    )

    print("\nOVERALL PERFORMANCE")
    print("-" * 70)

    print_summary(report.summary)

    print("\nCATEGORY-WISE PERFORMANCE")
    print("-" * 70)

    for category, summary in sorted(
        report.category_summaries.items()
    ):

        print(f"\nCategory: {category}")

        print_summary(summary)

    print("\nMISSED RELEVANT DOCUMENTS")
    print("-" * 70)

    missed_cases = [
        result
        for result in report.results
        if result.recall_at_k is not None
        and result.recall_at_k < 1.0
    ]

    if not missed_cases:

        print("No missed relevant documents.")

    for result in missed_cases:

        missing = (
            set(result.expected_documents)
            - set(result.retrieved_documents)
        )

        print(f"\nCase: {result.case_id}")

        print(f"Question: {result.question}")

        print(
            f"Expected: "
            f"{result.expected_documents}"
        )

        print(
            f"Retrieved: "
            f"{result.retrieved_documents}"
        )

        print(f"Missing: {sorted(missing)}")

        print(
            f"Recall: "
            f"{format_score(result.recall_at_k)}"
        )

    print("\nUNEXPECTED RETRIEVALS")
    print("-" * 70)

    unexpected_cases = [
        result
        for result in report.results
        if not result.expected_documents
        and result.retrieved_documents
    ]

    if not unexpected_cases:

        print("No unexpected retrievals.")

    for result in unexpected_cases:

        print(f"\nCase: {result.case_id}")

        print(f"Question: {result.question}")

        for document, score in zip(
            result.retrieved_documents,
            result.similarity_scores
        ):

            print(f"  {document}: {score:.4f}")


def save_report(
    report: RetrievalEvaluationReport
) -> Path:

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    filename = (
        f"retrieval_k{report.top_k}"
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
        description="Evaluate RAG retrieval performance"
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

    evaluator = RetrievalEvaluator()

    report = evaluator.evaluate(
        cases=cases,
        top_k=args.top_k,
        min_similarity_score=args.threshold
    )

    print_report(report)

    report_path = save_report(report)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":

    main()