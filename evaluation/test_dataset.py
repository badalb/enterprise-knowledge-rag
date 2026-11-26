from collections import Counter
from pathlib import Path

from evaluation.dataset_loader import load_dataset


def main():

    cases = load_dataset()

    print("=" * 60)
    print("GOLDEN EVALUATION DATASET")
    print("=" * 60)

    print(f"Total cases: {len(cases)}")

    answerable = sum(
        case.answerable for case in cases
    )

    print(f"Answerable: {answerable}")

    print(
        f"Unanswerable: {len(cases) - answerable}"
    )

    print("\nCategory distribution:")

    categories = Counter(
        case.category for case in cases
    )

    for category, count in sorted(categories.items()):

        print(f"{category}: {count}")

    document_directory = Path(
        "data/sample_documents"
    )

    available_documents = {
        path.name
        for path in document_directory.iterdir()
        if path.is_file()
    }

    expected_documents = {
        document
        for case in cases
        for document in case.expected_documents
    }

    missing_documents = (
        expected_documents - available_documents
    )

    if missing_documents:

        raise ValueError(
            f"Missing documents: {missing_documents}"
        )

    print("\nDocument references validated successfully.")

    print("\nDataset loaded successfully.")


if __name__ == "__main__":

    main()