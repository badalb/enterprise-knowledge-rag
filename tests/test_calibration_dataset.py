import json

import pytest

from pydantic import ValidationError

from evaluation.calibration_models import (
    ScoreRange,
    JudgeCalibrationCase
)

from evaluation.calibration_dataset_loader import (
    load_calibration_dataset
)


def test_score_range_contains():

    score_range = ScoreRange(
        minimum=0.5,
        maximum=0.75
    )

    assert score_range.contains(0.5)

    assert score_range.contains(0.75)

    assert not score_range.contains(0.25)

    assert not score_range.contains(1.0)


def test_invalid_score_range():

    with pytest.raises(ValidationError):

        ScoreRange(
            minimum=0.75,
            maximum=0.5
        )


def test_score_outside_allowed_range():

    with pytest.raises(ValidationError):

        ScoreRange(
            minimum=0.0,
            maximum=1.5
        )


def test_load_calibration_dataset():

    cases = load_calibration_dataset()

    assert len(cases) == 12


def test_calibration_categories():

    cases = load_calibration_dataset()

    categories = {
        case.category for case in cases
    }

    assert len(categories) == 12


def test_calibration_case_ids_unique():

    cases = load_calibration_dataset()

    case_ids = [
        case.id for case in cases
    ]

    assert len(case_ids) == len(set(case_ids))


def test_invalid_expected_fact_id():

    case = load_calibration_dataset()[0]

    data = case.model_dump()

    data["expected_assessment"][
        "covered_fact_ids"
    ] = [99]

    with pytest.raises(ValidationError):

        JudgeCalibrationCase.model_validate(data)


def test_duplicate_expected_fact_ids():

    case = load_calibration_dataset()[0]

    data = case.model_dump()

    data["expected_assessment"][
        "covered_fact_ids"
    ] = [1, 1]

    with pytest.raises(ValidationError):

        JudgeCalibrationCase.model_validate(data)


def test_unanswerable_case_correctness():

    case = next(
        case
        for case in load_calibration_dataset()
        if not case.evaluation_case.answerable
    )

    data = case.model_dump()

    data["expected_assessment"]["correctness"] = {
        "minimum": 0.5,
        "maximum": 1.0
    }

    with pytest.raises(ValidationError):

        JudgeCalibrationCase.model_validate(data)


def test_missing_dataset(tmp_path):

    with pytest.raises(FileNotFoundError):

        load_calibration_dataset(
            tmp_path / "missing.json"
        )


def test_empty_dataset(tmp_path):

    path = tmp_path / "empty.json"

    path.write_text(
        "[]",
        encoding="utf-8"
    )

    with pytest.raises(ValueError):

        load_calibration_dataset(path)


def test_duplicate_calibration_case(tmp_path):

    case = load_calibration_dataset()[0]

    data = case.model_dump()

    path = tmp_path / "duplicate.json"

    path.write_text(
        json.dumps([data, data]),
        encoding="utf-8"
    )

    with pytest.raises(
        ValueError,
        match="Duplicate calibration case IDs"
    ):

        load_calibration_dataset(path)