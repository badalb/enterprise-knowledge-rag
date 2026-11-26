from collections.abc import Sequence


def precision_at_k(
    retrieved_documents: Sequence[str],
    expected_documents: Sequence[str],
    k: int
) -> float | None:

    if k <= 0:
        raise ValueError("k must be positive")

    expected = set(expected_documents)

    if not expected:
        return None

    retrieved = retrieved_documents[:k]

    relevant = set(retrieved) & expected

    return len(relevant) / k


def recall_at_k(
    retrieved_documents: Sequence[str],
    expected_documents: Sequence[str],
    k: int
) -> float | None:

    if k <= 0:
        raise ValueError("k must be positive")

    expected = set(expected_documents)

    if not expected:
        return None

    retrieved = retrieved_documents[:k]

    relevant = set(retrieved) & expected

    return len(relevant) / len(expected)


def reciprocal_rank(
    retrieved_documents: Sequence[str],
    expected_documents: Sequence[str]
) -> float | None:

    expected = set(expected_documents)

    if not expected:
        return None

    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):

        if document in expected:

            return 1.0 / rank

    return 0.0


def hit_at_k(
    retrieved_documents: Sequence[str],
    expected_documents: Sequence[str],
    k: int
) -> bool | None:

    if k <= 0:
        raise ValueError("k must be positive")

    expected = set(expected_documents)

    if not expected:
        return None

    retrieved = retrieved_documents[:k]

    return bool(set(retrieved) & expected)


def context_precision(
    retrieved_documents: Sequence[str],
    expected_documents: Sequence[str]
) -> float | None:

    expected = set(expected_documents)

    if not expected:
        return None

    if not retrieved_documents:
        return 0.0

    relevant_count = sum(
        document in expected
        for document in retrieved_documents
    )

    return relevant_count / len(retrieved_documents)


def mean_metric(
    values: Sequence[float | None]
) -> float | None:

    valid_values = [
        value
        for value in values
        if value is not None
    ]

    if not valid_values:
        return None

    return sum(valid_values) / len(valid_values)