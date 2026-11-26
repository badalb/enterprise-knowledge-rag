from typing import Literal

from pydantic import BaseModel, Field, model_validator

from evaluation.models import EvaluationCase


CalibrationCategory = Literal[
    "fully_correct",
    "incomplete_answer",
    "additional_supported_information",
    "unsupported_claim",
    "contradictory_claim",
    "incorrect_citation",
    "irrelevant_information",
    "missing_context",
    "numerical_error",
    "unsupported_causal_inference",
    "multi_document",
    "unanswerable_question"
]


class ScoreRange(BaseModel):

    minimum: float = Field(ge=0.0, le=1.0)

    maximum: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_range(self):

        if self.minimum > self.maximum:

            raise ValueError(
                "Minimum score cannot exceed maximum score"
            )

        return self

    def contains(self, score: float) -> bool:

        return self.minimum <= score <= self.maximum


class CalibrationSource(BaseModel):

    filename: str = Field(min_length=1)

    page_number: int | None = None

    content: str = Field(min_length=1)


class ExpectedJudgeAssessment(BaseModel):

    correctness: ScoreRange

    faithfulness: ScoreRange

    answer_relevance: ScoreRange

    citation_support: ScoreRange

    covered_fact_ids: list[int] = Field(
        default_factory=list
    )

    has_unsupported_claims: bool

    has_contradictory_claims: bool

    has_citation_issues: bool


class JudgeCalibrationCase(BaseModel):

    id: str = Field(min_length=1)

    category: CalibrationCategory

    description: str = Field(min_length=1)

    evaluation_case: EvaluationCase

    retrieved_context: list[CalibrationSource] = Field(
        min_length=1
    )

    candidate_answer: str = Field(min_length=1)

    expected_assessment: ExpectedJudgeAssessment

    @model_validator(mode="after")
    def validate_calibration_case(self):

        expected_facts = (
            self.evaluation_case.expected_facts
        )

        fact_ids = (
            self.expected_assessment.covered_fact_ids
        )

        if len(fact_ids) != len(set(fact_ids)):

            raise ValueError(
                "Duplicate covered fact IDs"
            )

        for fact_id in fact_ids:

            if not 1 <= fact_id <= len(expected_facts):

                raise ValueError(
                    f"Invalid expected fact ID: {fact_id}"
                )

        if not self.evaluation_case.answerable:

            if fact_ids:

                raise ValueError(
                    "Unanswerable cases cannot cover "
                    "expected facts"
                )

            correctness = (
                self.expected_assessment.correctness
            )

            if (
                correctness.minimum != 0.0
                or correctness.maximum != 0.0
            ):

                raise ValueError(
                    "Fabricated answers to unanswerable "
                    "questions must have zero correctness"
                )

        return self