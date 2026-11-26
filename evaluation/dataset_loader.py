import json

from pathlib import Path

from evaluation.models import EvaluationCase


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent
    / "datasets"
    / "golden_dataset.json"
)


def load_dataset(
    file_path: str | Path = DEFAULT_DATASET_PATH
) -> list[EvaluationCase]:

    path = Path(file_path)

    if not path.is_file():

        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:

        data = json.load(file)

    cases = [
        EvaluationCase.model_validate(item)
        for item in data
    ]

    ids = [case.id for case in cases]

    if len(ids) != len(set(ids)):

        raise ValueError(
            "Duplicate evaluation case IDs"
        )

    return cases