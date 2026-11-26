import pytest

from app.models.chunk import RetrievedChunk

from app.models.execution_trace import (
    RAGExecutionTrace,
    RetrievalConfiguration
)

from app.models.query import (
    QueryResponse,
    QueryMetrics
)

from app.services.rag_service import RAGService

from evaluation.models import EvaluationCase

from evaluation.generation_metrics import (
    evaluate_deterministic,
    summarize_deterministic,
    safe_divide,
    percentile
)


ABSTENTION = RAGService.INSUFFICIENT_CONTEXT


def make_case(
    answerable: bool = True
) -> EvaluationCase:

    return EvaluationCase(
        id="TEST-001",

        question="Test question",

        expected_documents=(
            ["test.md"] if answerable else []
        ),

        answerable=answerable,

        category=(
            "direct"
            if answerable
            else "out_of_domain"
        )
    )


def make_trace(
    raw_answer: str | None = "Test answer [1].",
    abstained: bool = False,
    reason=None,
    generation_attempted: bool = True,
    invalid_citations: bool = False,
    chunk_count: int = 1
) -> RAGExecutionTrace:

    chunks = [
        RetrievedChunk(
            chunk_id=f"CHUNK-{index}",

            document_id="DOC-001",

            filename="test.md",

            page_number=None,

            chunk_index=index,

            content="Test document content.",

            similarity_score=0.75
        )
        for index in range(chunk_count)
    ]

    response = QueryResponse(
        question="Test question",

        answer=(
            ABSTENTION
            if abstained
            else raw_answer or ""
        ),

        sources=[],

        metrics=QueryMetrics(
            retrieved_chunks=chunk_count,

            retrieval_latency_ms=100.0,

            generation_latency_ms=(
                500.0
                if generation_attempted
                else 0.0
            ),

            total_latency_ms=(
                600.0
                if generation_attempted
                else 100.0
            ),

            input_tokens=(
                200
                if generation_attempted
                else 0
            ),

            output_tokens=(
                50
                if generation_attempted
                else 0
            )
        )
    )

    return RAGExecutionTrace(
        question="Test question",

        retrieval_configuration=(
            RetrievalConfiguration(
                top_k=2,
                min_similarity_score=0.30
            )
        ),

        retrieved_chunks=chunks,

        generation_attempted=generation_attempted,

        raw_answer=raw_answer,

        invalid_citations=invalid_citations,

        abstained=abstained,

        abstention_reason=reason,

        response=response
    )


def test_correct_answer():

    result = evaluate_deterministic(
        make_case(answerable=True),
        make_trace()
    )

    assert result.abstention_correct is True

    assert result.citation_valid is True

    assert result.citation_present is True

    assert result.guardrail_intervened is False


def test_false_abstention():

    result = evaluate_deterministic(
        make_case(answerable=True),

        make_trace(
            raw_answer=ABSTENTION,
            abstained=True,
            reason="MODEL_ABSTENTION"
        )
    )

    assert result.abstention_correct is False

    assert result.model_abstained is True

    assert result.citation_valid is None


def test_correct_model_abstention():

    result = evaluate_deterministic(
        make_case(answerable=False),

        make_trace(
            raw_answer=ABSTENTION,
            abstained=True,
            reason="MODEL_ABSTENTION"
        )
    )

    assert result.abstention_correct is True

    assert result.model_abstained is True

    assert result.citation_present is None


def test_no_retrieved_context():

    result = evaluate_deterministic(
        make_case(answerable=False),

        make_trace(
            raw_answer=None,
            abstained=True,
            reason="NO_RETRIEVED_CONTEXT",
            generation_attempted=False,
            chunk_count=0
        )
    )

    assert result.abstention_correct is True

    assert result.generation_attempted is False

    assert result.model_abstained is False

    assert result.citation_valid is None

    assert result.input_tokens == 0


def test_invalid_citation():

    result = evaluate_deterministic(
        make_case(answerable=True),

        make_trace(
            raw_answer="Test answer [9].",
            abstained=True,
            reason="INVALID_CITATIONS",
            invalid_citations=True
        )
    )

    assert result.citation_valid is False

    assert result.citation_present is True

    assert result.guardrail_intervened is True

    assert result.abstention_correct is False


def test_missing_citation():

    result = evaluate_deterministic(
        make_case(answerable=True),

        make_trace(
            raw_answer="Test answer.",
            abstained=True,
            reason="MISSING_CITATIONS"
        )
    )

    assert result.citation_present is False

    assert result.citation_valid is False

    assert result.guardrail_intervened is True


def test_incorrect_answer_to_unanswerable_question():

    result = evaluate_deterministic(
        make_case(answerable=False),
        make_trace()
    )

    assert result.abstention_correct is False

    assert result.actual_abstained is False

    assert result.citation_valid is True


def test_summary():

    results = [
        evaluate_deterministic(
            make_case(True),
            make_trace()
        ),

        evaluate_deterministic(
            make_case(True),

            make_trace(
                raw_answer=ABSTENTION,
                abstained=True,
                reason="MODEL_ABSTENTION"
            )
        ),

        evaluate_deterministic(
            make_case(False),

            make_trace(
                raw_answer=None,
                abstained=True,
                reason="NO_RETRIEVED_CONTEXT",
                generation_attempted=False,
                chunk_count=0
            )
        ),

        evaluate_deterministic(
            make_case(False),
            make_trace()
        )
    ]

    summary = summarize_deterministic(results)

    assert summary.total_cases == 4

    assert summary.true_positives == 1

    assert summary.true_negatives == 1

    assert summary.false_positives == 1

    assert summary.false_negatives == 1

    assert summary.abstention_accuracy == 0.5

    assert summary.answerable_response_rate == 0.5

    assert summary.false_abstention_rate == 0.5

    assert summary.unanswerable_rejection_rate == 0.5

    assert summary.generation_invocation_rate == 0.75

    assert summary.citation_validity_rate == 1.0

    assert summary.model_abstention_rate == pytest.approx(
        1 / 3
    )

    assert summary.total_input_tokens == 600

    assert summary.total_output_tokens == 150

    assert summary.mean_input_tokens == 150

    assert summary.mean_generation_latency_ms == 500

    assert summary.mean_total_latency_ms == 475


def test_summary_without_generation():

    result = evaluate_deterministic(
        make_case(False),

        make_trace(
            raw_answer=None,
            abstained=True,
            reason="NO_RETRIEVED_CONTEXT",
            generation_attempted=False,
            chunk_count=0
        )
    )

    summary = summarize_deterministic([result])

    assert summary.abstention_accuracy == 1.0

    assert summary.generation_invocation_rate == 0.0

    assert summary.citation_validity_rate is None

    assert summary.model_abstention_rate is None

    assert summary.guardrail_intervention_rate is None

    assert summary.mean_generation_latency_ms is None


def test_empty_summary():

    with pytest.raises(ValueError):

        summarize_deterministic([])


def test_safe_divide():

    assert safe_divide(1, 2) == 0.5

    assert safe_divide(1, 0) is None


def test_percentile():

    assert percentile(
        [100, 200, 300, 400, 500],
        95
    ) == 500


def test_missing_token_usage():

    result = evaluate_deterministic(
        make_case(True),
        make_trace()
    )

    result.input_tokens = None

    summary = summarize_deterministic([result])

    assert summary.total_input_tokens is None

    assert summary.mean_input_tokens is None