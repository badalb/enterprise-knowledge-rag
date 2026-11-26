from collections import Counter

from evaluation.calibration_dataset_loader import (
    load_calibration_dataset
)

from app.generation.citation_parser import (
    CitationParser
)


def main():

    cases = load_calibration_dataset()

    categories = Counter(
        case.category for case in cases
    )

    print("=" * 70)

    print("JUDGE CALIBRATION DATASET VALIDATION")

    print("=" * 70)

    print(f"Total calibration cases: {len(cases)}")

    print(f"Calibration categories: {len(categories)}")

    print(
        "Answerable cases: "
        f"{sum(c.evaluation_case.answerable for c in cases)}"
    )

    print(
        "Unanswerable cases: "
        f"{sum(not c.evaluation_case.answerable for c in cases)}"
    )

    print("\nCATEGORY DISTRIBUTION")

    for category, count in sorted(categories.items()):

        print(f"{category}: {count}")

    print("\nCASE VALIDATION")

    for case in cases:

        citation_ids = CitationParser.extract_ids(
            case.candidate_answer
        )

        if not citation_ids:

            raise ValueError(
                f"Missing candidate citations: {case.id}"
            )

        invalid_citations = (
            CitationParser.has_invalid_citations(
                answer=case.candidate_answer,
                chunk_count=len(case.retrieved_context)
            )
        )

        if invalid_citations:

            raise ValueError(
                f"Out-of-range citations: {case.id}"
            )

        print(
            f"{case.id} | "
            f"{case.category} | "
            f"Sources: {len(case.retrieved_context)} | "
            f"Expected facts: "
            f"{len(case.evaluation_case.expected_facts)} | "
            f"VALID"
        )

    print("\nAll calibration cases validated successfully.")


if __name__ == "__main__":

    main()