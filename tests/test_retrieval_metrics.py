import pytest

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_at_k,
    context_precision,
    mean_metric
)


EXPECTED = [
    "kafka.md",
    "incident.md"
]


RETRIEVED = [
    "kafka.md",
    "spark.md",
    "incident.md",
    "database.md",
    "deployment.md"
]


def test_precision_at_k():

    assert precision_at_k(
        RETRIEVED, EXPECTED, 5
    ) == pytest.approx(0.4)


def test_recall_at_k():

    assert recall_at_k(
        RETRIEVED, EXPECTED, 5
    ) == pytest.approx(1.0)


def test_reciprocal_rank():

    assert reciprocal_rank(
        RETRIEVED, EXPECTED
    ) == pytest.approx(1.0)


def test_hit_at_k():

    assert hit_at_k(
        RETRIEVED, EXPECTED, 5
    ) is True


def test_context_precision():

    assert context_precision(
        RETRIEVED, EXPECTED
    ) == pytest.approx(0.4)


def test_partial_recall():

    assert recall_at_k(
        RETRIEVED, EXPECTED, 2
    ) == pytest.approx(0.5)


def test_first_relevant_at_rank_three():

    retrieved = [
        "spark.md",
        "database.md",
        "kafka.md"
    ]

    assert reciprocal_rank(
        retrieved, EXPECTED
    ) == pytest.approx(1 / 3)


def test_no_relevant_results():

    retrieved = [
        "spark.md",
        "database.md"
    ]

    assert precision_at_k(
        retrieved, EXPECTED, 2
    ) == 0.0

    assert recall_at_k(
        retrieved, EXPECTED, 2
    ) == 0.0

    assert reciprocal_rank(
        retrieved, EXPECTED
    ) == 0.0

    assert hit_at_k(
        retrieved, EXPECTED, 2
    ) is False


def test_empty_retrieval():

    assert precision_at_k(
        [], EXPECTED, 5
    ) == 0.0

    assert recall_at_k(
        [], EXPECTED, 5
    ) == 0.0

    assert reciprocal_rank(
        [], EXPECTED
    ) == 0.0

    assert hit_at_k(
        [], EXPECTED, 5
    ) is False

    assert context_precision(
        [], EXPECTED
    ) == 0.0


def test_no_expected_documents():

    assert precision_at_k(
        RETRIEVED, [], 5
    ) is None

    assert recall_at_k(
        RETRIEVED, [], 5
    ) is None

    assert reciprocal_rank(
        RETRIEVED, []
    ) is None

    assert hit_at_k(
        RETRIEVED, [], 5
    ) is None

    assert context_precision(
        RETRIEVED, []
    ) is None


def test_duplicate_documents():

    retrieved = [
        "kafka.md",
        "kafka.md",
        "incident.md"
    ]

    assert precision_at_k(
        retrieved, EXPECTED, 3
    ) == pytest.approx(2 / 3)

    assert recall_at_k(
        retrieved, EXPECTED, 3
    ) == pytest.approx(1.0)

    assert context_precision(
        retrieved, EXPECTED
    ) == pytest.approx(1.0)


def test_threshold_filtered_results():

    retrieved = [
        "kafka.md",
        "incident.md"
    ]

    assert precision_at_k(
        retrieved, EXPECTED, 5
    ) == pytest.approx(0.4)

    assert recall_at_k(
        retrieved, EXPECTED, 5
    ) == pytest.approx(1.0)

    assert context_precision(
        retrieved, EXPECTED
    ) == pytest.approx(1.0)


def test_mean_metric():

    assert mean_metric(
        [1.0, 0.5, None]
    ) == pytest.approx(0.75)


def test_mean_metric_no_valid_values():

    assert mean_metric(
        [None, None]
    ) is None

    assert mean_metric([]) is None


@pytest.mark.parametrize(
    "metric",
    [
        precision_at_k,
        recall_at_k,
        hit_at_k
    ]
)
def test_invalid_k(metric):

    with pytest.raises(ValueError):

        metric(RETRIEVED, EXPECTED, 0)