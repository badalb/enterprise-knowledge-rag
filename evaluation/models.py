from typing import Literal

from pydantic import BaseModel, Field

from app.models.execution_trace import (
    RAGExecutionTrace,
    AbstentionReason
)



EvaluationCategory = Literal[
    "direct",
    "semantic",
    "multi_document",
    "numerical",
    "in_domain_unanswerable",
    "out_of_domain",
    "adversarial"
]


class EvaluationCase(BaseModel):

    id: str

    question: str = Field(min_length=1)

    expected_documents: list[str] = Field(
        default_factory=list
    )

    answerable: bool

    category: EvaluationCategory

    expected_answer: str | None = None

    expected_facts: list[str] = Field(
        default_factory=list
    )


class RetrievalEvaluationResult(BaseModel):

    case_id: str

    question: str

    category: EvaluationCategory

    expected_documents: list[str]

    retrieved_documents: list[str]

    similarity_scores: list[float]

    top_k: int

    precision_at_k: float | None = None

    recall_at_k: float | None = None

    reciprocal_rank: float | None = None

    hit: bool | None = None

    latency_ms: float

    context_precision: float | None = None

    min_similarity_score: float = Field(
    default=0.25,
    ge=-1.0,
    le=1.0
)

class RetrievalEvaluationSummary(BaseModel):

    total_cases: int

    evaluated_cases: int

    no_expected_document_cases: int

    mean_precision_at_k: float | None = None

    mean_recall_at_k: float | None = None

    mean_reciprocal_rank: float | None = None

    hit_rate: float | None = None

    mean_context_precision: float | None = None

    mean_retrieved_chunks: float

    mean_latency_ms: float

    p95_latency_ms: float

    no_relevant_document_rejection_rate: float | None = None


class RetrievalEvaluationReport(BaseModel):

    evaluated_at: str

    embedding_model: str

    collection_name: str

    indexed_chunks: int

    top_k: int

    min_similarity_score: float

    summary: RetrievalEvaluationSummary

    category_summaries: dict[
        str, RetrievalEvaluationSummary
    ]

    results: list[RetrievalEvaluationResult]

class RetrievalExperimentResult(BaseModel):

    top_k: int

    min_similarity_score: float

    mean_precision_at_k: float | None = None

    mean_recall_at_k: float | None = None

    mean_reciprocal_rank: float | None = None

    hit_rate: float | None = None

    mean_context_precision: float | None = None

    retrieval_rejection_rate: float | None = None

    mean_retrieved_chunks: float

    full_recall_rate: float | None = None


class RetrievalExperimentReport(BaseModel):

    evaluated_at: str

    embedding_model: str

    indexed_chunks: int

    total_cases: int

    total_configurations: int

    candidate_retrieval_latency_ms: float

    baseline: RetrievalExperimentResult

    recommended: RetrievalExperimentResult | None = None

    results: list[RetrievalExperimentResult]


class DeterministicEvaluationResult(BaseModel):

    case_id: str

    category: EvaluationCategory

    expected_answerable: bool

    actual_abstained: bool

    abstention_correct: bool

    abstention_reason: AbstentionReason | None = None

    generation_attempted: bool

    model_abstained: bool

    guardrail_intervened: bool

    citation_valid: bool | None = None

    citation_present: bool | None = None

    retrieved_chunks: int

    input_tokens: int | None = None

    output_tokens: int | None = None

    retrieval_latency_ms: float

    generation_latency_ms: float

    total_latency_ms: float


class DeterministicEvaluationSummary(BaseModel):

    total_cases: int

    answerable_cases: int

    unanswerable_cases: int

    true_positives: int

    true_negatives: int

    false_positives: int

    false_negatives: int

    abstention_accuracy: float

    answerable_response_rate: float | None = None

    false_abstention_rate: float | None = None

    unanswerable_rejection_rate: float | None = None

    citation_validity_rate: float | None = None

    citation_coverage_rate: float | None = None

    generation_invocation_rate: float

    model_abstention_rate: float | None = None

    guardrail_intervention_rate: float | None = None

    abstention_reason_counts: dict[str, int]

    mean_retrieved_chunks: float

    total_input_tokens: int | None = None

    total_output_tokens: int | None = None

    mean_input_tokens: float | None = None

    mean_output_tokens: float | None = None

    mean_retrieval_latency_ms: float

    mean_generation_latency_ms: float | None = None

    mean_total_latency_ms: float

    p95_total_latency_ms: float


JudgeScore = Literal[
    0.0,
    0.25,
    0.50,
    0.75,
    1.0
]


class JudgeAssessment(BaseModel):

    correctness: JudgeScore

    faithfulness: JudgeScore

    answer_relevance: JudgeScore

    citation_support: JudgeScore

    covered_fact_ids: list[int]

    unsupported_claims: list[str]

    contradictory_claims: list[str]

    citation_issues: list[str]

    reasoning: str


class SemanticEvaluationResult(BaseModel):

    case_id: str

    judge_model: str

    correctness: float

    faithfulness: float

    answer_relevance: float

    citation_support: float

    fact_coverage: float | None = None

    covered_facts: list[str]

    missing_facts: list[str]

    unsupported_claims: list[str]

    contradictory_claims: list[str]

    citation_issues: list[str]

    reasoning: str

    judge_input_tokens: int | None = None

    judge_output_tokens: int | None = None

    judge_latency_ms: float

class GenerationEvaluationResult(BaseModel):

    case: EvaluationCase

    trace: RAGExecutionTrace

    deterministic: DeterministicEvaluationResult

    semantic: SemanticEvaluationResult | None = None


class SemanticEvaluationSummary(BaseModel):

    total_cases: int

    judged_cases: int

    skipped_cases: int

    judged_answerable_cases: int

    mean_correctness: float | None = None

    mean_faithfulness: float | None = None

    mean_answer_relevance: float | None = None

    mean_citation_support: float | None = None

    mean_fact_coverage: float | None = None

    end_to_end_correctness: float | None = None

    end_to_end_fact_coverage: float | None = None

    total_expected_facts: int

    total_covered_facts: int

    micro_fact_coverage: float | None = None

    unsupported_claim_count: int

    contradictory_claim_count: int

    citation_issue_count: int

    total_judge_input_tokens: int | None = None

    total_judge_output_tokens: int | None = None

    mean_judge_latency_ms: float | None = None


class GenerationEvaluationSummary(BaseModel):

    deterministic: DeterministicEvaluationSummary

    semantic: SemanticEvaluationSummary


class GenerationEvaluationReport(BaseModel):

    evaluated_at: str

    embedding_model: str

    generation_model: str

    judge_model: str

    collection_name: str

    indexed_chunks: int

    top_k: int

    min_similarity_score: float

    evaluation_duration_ms: float

    summary: GenerationEvaluationSummary

    category_summaries: dict[
        str, GenerationEvaluationSummary
    ]

    results: list[GenerationEvaluationResult]