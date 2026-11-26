import math

from collections import defaultdict
from datetime import datetime, timezone
from time import perf_counter

from app.config import get_settings
from app.retrieval.vector_retriever import VectorRetriever

from evaluation.models import (
    EvaluationCase,
    RetrievalEvaluationResult,
    RetrievalEvaluationSummary,
    RetrievalEvaluationReport
)

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_at_k,
    context_precision,
    mean_metric
)


class RetrievalEvaluator:

    def __init__(
        self,
        retriever: VectorRetriever | None = None
    ):

        self.settings = get_settings()

        self.retriever = (
            retriever
            if retriever is not None
            else VectorRetriever()
        )

    def evaluate_case(
        self,
        case: EvaluationCase,
        top_k: int,
        min_similarity_score: float
    ) -> RetrievalEvaluationResult:

        start = perf_counter()

        chunks = self.retriever.retrieve(
            question=case.question,
            top_k=top_k,
            min_similarity_score=min_similarity_score
        )

        latency_ms = (
            perf_counter() - start
        ) * 1000

        retrieved_documents = [
            chunk.filename
            for chunk in chunks
        ]

        similarity_scores = [
            chunk.similarity_score
            for chunk in chunks
        ]

        expected_documents = case.expected_documents

        return RetrievalEvaluationResult(
            case_id=case.id,

            question=case.question,

            category=case.category,

            expected_documents=expected_documents,

            retrieved_documents=retrieved_documents,

            similarity_scores=similarity_scores,

            top_k=top_k,

            min_similarity_score=min_similarity_score,

            precision_at_k=precision_at_k(
                retrieved_documents,
                expected_documents,
                top_k
            ),

            recall_at_k=recall_at_k(
                retrieved_documents,
                expected_documents,
                top_k
            ),

            reciprocal_rank=reciprocal_rank(
                retrieved_documents,
                expected_documents
            ),

            hit=hit_at_k(
                retrieved_documents,
                expected_documents,
                top_k
            ),

            context_precision=context_precision(
                retrieved_documents,
                expected_documents
            ),

            latency_ms=latency_ms
        )

    def evaluate(
        self,
        cases: list[EvaluationCase],
        top_k: int | None = None,
        min_similarity_score: float | None = None
    ) -> RetrievalEvaluationReport:

        if not cases:
            raise ValueError("Evaluation dataset is empty")

        if top_k is None:
            top_k = self.settings.top_k

        if min_similarity_score is None:
            min_similarity_score = (
                self.settings.min_similarity_score
            )

        indexed_chunks = (
            self.retriever.vector_store.count()
        )

        if indexed_chunks == 0:
            raise ValueError(
                "Vector collection is empty. "
                "Ingest documents before evaluation."
            )

        results = []

        for case in cases:

            print(
                f"Evaluating {case.id}: "
                f"{case.question}"
            )

            result = self.evaluate_case(
                case=case,
                top_k=top_k,
                min_similarity_score=min_similarity_score
            )

            results.append(result)

        grouped_results = defaultdict(list)

        for result in results:

            grouped_results[
                result.category
            ].append(result)

        category_summaries = {
            category: self.summarize(category_results)
            for category, category_results
            in grouped_results.items()
        }

        return RetrievalEvaluationReport(
            evaluated_at=datetime.now(
                timezone.utc
            ).isoformat(),

            embedding_model=(
                self.settings.openai_embedding_model
            ),

            collection_name=(
                self.settings.chroma_collection_name
            ),

            indexed_chunks=indexed_chunks,

            top_k=top_k,

            min_similarity_score=min_similarity_score,

            summary=self.summarize(results),

            category_summaries=category_summaries,

            results=results
        )

    @staticmethod
    def summarize(
        results: list[RetrievalEvaluationResult]
    ) -> RetrievalEvaluationSummary:

        if not results:
            raise ValueError("No results to summarize")

        applicable_results = [
            result
            for result in results
            if result.expected_documents
        ]

        no_expected_document_results = [
            result
            for result in results
            if not result.expected_documents
        ]

        hit_rate = mean_metric([
            float(result.hit)
            if result.hit is not None
            else None
            for result in results
        ])

        rejection_rate = mean_metric([
            float(not result.retrieved_documents)
            for result in no_expected_document_results
        ])

        latencies = sorted([
            result.latency_ms
            for result in results
        ])

        p95_index = (
            math.ceil(0.95 * len(latencies)) - 1
        )

        return RetrievalEvaluationSummary(
            total_cases=len(results),

            evaluated_cases=len(applicable_results),

            no_expected_document_cases=len(
                no_expected_document_results
            ),

            mean_precision_at_k=mean_metric([
                result.precision_at_k
                for result in results
            ]),

            mean_recall_at_k=mean_metric([
                result.recall_at_k
                for result in results
            ]),

            mean_reciprocal_rank=mean_metric([
                result.reciprocal_rank
                for result in results
            ]),

            hit_rate=hit_rate,

            mean_context_precision=mean_metric([
                result.context_precision
                for result in results
            ]),

            mean_retrieved_chunks=(
                sum(
                    len(result.retrieved_documents)
                    for result in results
                ) / len(results)
            ),

            mean_latency_ms=(
                sum(latencies) / len(latencies)
            ),

            p95_latency_ms=latencies[p95_index],

            no_relevant_document_rejection_rate=(
                rejection_rate
            )
        )