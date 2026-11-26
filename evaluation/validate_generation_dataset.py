from collections import Counter

from evaluation.dataset_loader import load_dataset


ABSTENTION_RESPONSE = (
    "I could not find sufficient information "
    "in the available documents to answer this question."
)


def main():

    cases = load_dataset()

    errors = []

    category_counts = Counter()

    answerable_count = 0

    unanswerable_count = 0

    total_expected_facts = 0

    for case in cases:

        category_counts[case.category] += 1

        if not case.expected_answer or not case.expected_answer.strip():

            errors.append(
                f"{case.id}: Missing expected answer"
            )

        if case.answerable:

            answerable_count += 1

            if not case.expected_documents:

                errors.append(
                    f"{case.id}: Missing expected documents"
                )

            if not case.expected_facts:

                errors.append(
                    f"{case.id}: Missing expected facts"
                )

            if case.expected_answer == ABSTENTION_RESPONSE:

                errors.append(
                    f"{case.id}: Answerable case has "
                    "an abstention reference"
                )

            total_expected_facts += len(
                case.expected_facts
            )

        else:

            unanswerable_count += 1

            if case.expected_answer != ABSTENTION_RESPONSE:

                errors.append(
                    f"{case.id}: Invalid abstention response"
                )

            if case.expected_facts:

                errors.append(
                    f"{case.id}: Unanswerable case "
                    "contains expected facts"
                )

        if any(
            not fact.strip()
            for fact in case.expected_facts
        ):

            errors.append(
                f"{case.id}: Empty expected fact"
            )

    print("=" * 60)

    print("GENERATION DATASET VALIDATION")

    print("=" * 60)

    print(f"Total cases: {len(cases)}")

    print(f"Answerable cases: {answerable_count}")

    print(f"Unanswerable cases: {unanswerable_count}")

    print(f"Total expected facts: {total_expected_facts}")

    print("\nCATEGORY DISTRIBUTION")

    for category, count in sorted(
        category_counts.items()
    ):

        print(f"{category}: {count}")

    if errors:

        print("\nVALIDATION ERRORS")

        for error in errors:

            print(f"- {error}")

        raise ValueError(
            f"Dataset validation failed: "
            f"{len(errors)} errors"
        )

    print("\nGeneration dataset validation successful.")


if __name__ == "__main__":

    main()