import pytest

from evaluation.models import (
    EvaluationCase,
    JudgeAssessment
)

from evaluation.llm_judge import LLMJudge


def make_case(
    answerable: bool = True
) -> EvaluationCase:

    return EvaluationCase(
        id="TEST-001",

        question="What are the RPO and RTO targets?",

        expected_documents=[
            "disaster_recovery.md"
        ],

        answerable=answerable,

        category=(
            "numerical"
            if answerable
            else "in_domain_unanswerable"
        ),

        expected_answer=(
            "RPO is 15 minutes and RTO is 60 minutes."
            if answerable
            else "Insufficient information."
        ),

        expected_facts=(
            [
                "RPO is 15 minutes.",
                "RTO is 60 minutes."
            ]
            if answerable
            else []
        )
    )


def make_assessment(
    covered_fact_ids: list[int],
    correctness: float = 1.0
) -> JudgeAssessment:

    return JudgeAssessment(
        correctness=correctness,

        faithfulness=1.0,

        answer_relevance=1.0,

        citation_support=1.0,

        covered_fact_ids=covered_fact_ids,

        unsupported_claims=[],

        contradictory_claims=[],

        citation_issues=[],

        reasoning="Test assessment."
    )


def build_result(case, assessment):

    return LLMJudge.build_result(
        case=case,

        assessment=assessment,

        judge_model="test-model",

        judge_latency_ms=100.0,

        input_tokens=500,

        output_tokens=100
    )


def test_full_fact_coverage():

    result = build_result(
        make_case(),
        make_assessment([1, 2])
    )

    assert result.fact_coverage == 1.0

    assert len(result.covered_facts) == 2

    assert result.missing_facts == []


def test_partial_fact_coverage():

    result = build_result(
        make_case(),
        make_assessment([1], correctness=0.5)
    )

    assert result.fact_coverage == 0.5

    assert result.covered_facts == [
        "RPO is 15 minutes."
    ]

    assert result.missing_facts == [
        "RTO is 60 minutes."
    ]


def test_zero_fact_coverage():

    result = build_result(
        make_case(),
        make_assessment([], correctness=0.0)
    )

    assert result.fact_coverage == 0.0

    assert len(result.missing_facts) == 2


def test_duplicate_fact_ids():

    result = build_result(
        make_case(),
        make_assessment([1, 1, 1])
    )

    assert result.fact_coverage == 0.5

    assert len(result.covered_facts) == 1


def test_invalid_fact_id():

    with pytest.raises(
        ValueError,
        match="Invalid expected fact IDs"
    ):

        build_result(
            make_case(),
            make_assessment([1, 99])
        )


def test_zero_fact_id():

    with pytest.raises(ValueError):

        build_result(
            make_case(),
            make_assessment([0])
        )


def test_unanswerable_question():

    result = build_result(
        make_case(answerable=False),
        make_assessment([], correctness=1.0)
    )

    assert result.correctness == 0.0

    assert result.fact_coverage is None


def test_judge_usage_metrics():

    result = build_result(
        make_case(),
        make_assessment([1, 2])
    )

    assert result.judge_input_tokens == 500

    assert result.judge_output_tokens == 100

    assert result.judge_latency_ms == 100.0

    assert result.judge_model == "test-model"


def test_invalid_judge_score():

    with pytest.raises(ValueError):

        make_assessment(
            [1, 2],
            correctness=0.83
        )
