import json

from pathlib import Path

from evaluation.calibration_models import (
    JudgeCalibrationCase
)


DEFAULT_CALIBRATION_DATASET_PATH = (
    Path(__file__).resolve().parent
    / "datasets"
    / "judge_calibration_dataset.json"
)


def load_calibration_dataset(
    path: Path | str = DEFAULT_CALIBRATION_DATASET_PATH
) -> list[JudgeCalibrationCase]:

    path = Path(path)

    if not path.is_file():

        raise FileNotFoundError(
            f"Calibration dataset not found: {path}"
        )

    raw_data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(raw_data, list):

        raise ValueError(
            "Calibration dataset must contain a JSON array"
        )

    if not raw_data:

        raise ValueError(
            "Calibration dataset cannot be empty"
        )

    cases = [
        JudgeCalibrationCase.model_validate(item)
        for item in raw_data
    ]

    case_ids = [case.id for case in cases]

    if len(case_ids) != len(set(case_ids)):

        raise ValueError(
            "Duplicate calibration case IDs"
        )

    for case in cases:

        if case.id != case.evaluation_case.id:

            raise ValueError(
                f"Calibration case ID mismatch: {case.id}"
            )

        evaluation_case = case.evaluation_case

        if not (
            evaluation_case.expected_answer or ""
        ).strip():

            raise ValueError(
                f"Missing reference answer: {case.id}"
            )

        if (
            evaluation_case.answerable
            and not evaluation_case.expected_facts
        ):

            raise ValueError(
                f"Missing expected facts: {case.id}"
            )

        if (
            not evaluation_case.answerable
            and evaluation_case.expected_facts
        ):

            raise ValueError(
                f"Unexpected facts for unanswerable "
                f"case: {case.id}"
            )

        if any(
            not fact.strip()
            for fact in evaluation_case.expected_facts
        ):

            raise ValueError(
                f"Blank expected fact: {case.id}"
            )

        if not case.candidate_answer.strip():

            raise ValueError(
                f"Blank candidate answer: {case.id}"
            )

        for source in case.retrieved_context:

            if not source.content.strip():

                raise ValueError(
                    f"Empty source content: {case.id}"
                )

    return cases