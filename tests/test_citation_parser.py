import pytest

from app.generation.citation_parser import CitationParser


@pytest.mark.parametrize(
    "raw_answer, expected",
    [
        ("Answer [1].", "Answer [1]."),

        ("Answer [SOURCE 1].", "Answer [1]."),

        ("Answer [source 1].", "Answer [1]."),

        ("Answer [Source 2].", "Answer [2]."),

        ("Answer [1, 2].", "Answer [1][2]."),

        (
            "Answer [SOURCE 1, SOURCE 2].",
            "Answer [1][2]."
        ),

        (
            "Answer [SOURCE 1, 2].",
            "Answer [1][2]."
        ),

        ("Answer [ 1 ].", "Answer [1]."),

        ("Answer [ABC].", "Answer [ABC]."),

        ("Answer without citations.",
         "Answer without citations.")
    ]
)
def test_normalize(raw_answer, expected):

    assert CitationParser.normalize(
        raw_answer
    ) == expected


def test_extract_single_citation():

    assert CitationParser.extract_ids(
        "Answer [SOURCE 1]."
    ) == [1]


def test_extract_multiple_citations():

    assert CitationParser.extract_ids(
        "Answer [SOURCE 1, 2] and [3]."
    ) == [1, 2, 3]


def test_extract_duplicate_citations():

    assert CitationParser.extract_ids(
        "Answer [1] and [SOURCE 1]."
    ) == [1, 1]


def test_extract_no_citations():

    assert CitationParser.extract_ids(
        "Answer without citations."
    ) == []


def test_valid_citations():

    assert not CitationParser.has_invalid_citations(
        answer="Answer [SOURCE 1, 2].",
        chunk_count=2
    )


def test_invalid_citation():

    assert CitationParser.has_invalid_citations(
        answer="Answer [SOURCE 99].",
        chunk_count=2
    )


def test_zero_citation():

    assert CitationParser.has_invalid_citations(
        answer="Answer [SOURCE 0].",
        chunk_count=2
    )


def test_mixed_valid_invalid_citations():

    assert CitationParser.has_invalid_citations(
        answer="Answer [1, 99].",
        chunk_count=2
    )


def test_citation_without_context():

    assert CitationParser.has_invalid_citations(
        answer="Answer [1].",
        chunk_count=0
    )


def test_normalization_is_idempotent():

    answer = "Answer [SOURCE 1, SOURCE 2]."

    normalized = CitationParser.normalize(answer)

    assert CitationParser.normalize(
        normalized
    ) == normalized