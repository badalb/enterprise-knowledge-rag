import argparse

from pathlib import Path

from app.generation.citation_parser import CitationParser

from app.generation.citation_builder import CitationBuilder

from evaluation.models import GenerationEvaluationReport


def main():

    parser = argparse.ArgumentParser(
        description="Replay historical citation failures"
    )

    parser.add_argument(
        "--report",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    report = GenerationEvaluationReport.model_validate_json(
        args.report.read_text(encoding="utf-8")
    )

    builder = CitationBuilder()

    total_failures = 0

    recovered = 0

    for result in report.results:

        if not result.deterministic.guardrail_intervened:

            continue

        total_failures += 1

        trace = result.trace

        raw_answer = trace.raw_answer or ""

        normalized_answer = CitationParser.normalize(
            raw_answer
        )

        sources = builder.build(
            answer=normalized_answer,
            chunks=trace.retrieved_chunks
        )

        invalid_citations = (
            builder.has_invalid_citations(
                answer=normalized_answer,
                chunk_count=len(trace.retrieved_chunks)
            )
        )

        recovered_case = (
            bool(sources)
            and not invalid_citations
        )

        if recovered_case:

            recovered += 1

        print("\n" + "=" * 70)

        print(f"CASE: {result.case.id}")

        print(f"QUESTION: {result.case.question}")

        print("\nORIGINAL ANSWER")

        print(raw_answer)

        print("\nNORMALIZED ANSWER")

        print(normalized_answer)

        print("\nRESOLVED CITATIONS")

        for source in sources:

            print(
                f"[{source.citation_id}] "
                f"{source.filename}"
            )

        print(
            f"\nInvalid citations: {invalid_citations}"
        )

        print(
            f"Citation validation recovered: "
            f"{recovered_case}"
        )

    print("\n" + "=" * 70)

    print("CITATION REGRESSION REPLAY SUMMARY")

    print(f"Historical failures: {total_failures}")

    print(f"Recovered: {recovered}")

    print(f"Remaining failures: {total_failures - recovered}")

    if total_failures:

        print(
            f"Recovery rate: "
            f"{recovered / total_failures:.2%}"
        )


if __name__ == "__main__":

    main()