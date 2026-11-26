import argparse
from pathlib import Path

from evaluation.models import GenerationEvaluationReport


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    report = GenerationEvaluationReport.model_validate_json(
        args.report.read_text(encoding="utf-8")
    )

    for result in report.results:

        if not result.deterministic.guardrail_intervened:

            continue

        trace = result.trace

        print("\n" + "=" * 70)

        print(f"CASE: {result.case.id}")

        print(f"QUESTION: {result.case.question}")

        print("\nRETRIEVED DOCUMENTS")

        for index, chunk in enumerate(
            trace.retrieved_chunks,
            start=1
        ):

            print(
                f"[{index}] {chunk.filename} "
                f"({chunk.similarity_score:.4f})"
            )

        print("\nRAW GENERATED ANSWER")

        print(repr(trace.raw_answer))

        print("\nFINAL APPLICATION ANSWER")

        print(trace.response.answer)

        print("\nABSTENTION REASON")

        print(trace.abstention_reason)


if __name__ == "__main__":

    main()