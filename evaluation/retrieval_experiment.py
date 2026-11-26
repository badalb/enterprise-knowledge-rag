from datetime import datetime, timezone
from itertools import product
from time import perf_counter

from app.config import get_settings
from app.models.chunk import RetrievedChunk
from app.retrieval.vector_retriever import VectorRetriever

from evaluation.models import (
    EvaluationCase,
    RetrievalExperimentResult,
    RetrievalExperimentReport
)

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_at_k,
    context_precision,
    mean_metric
)


class RetrievalExperiment:

    TOP_K_VALUES = [1, 2, 3, 5]

    THRESHOLD_VALUES = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45
    ]

    BASELINE_TOP_K = 5

    BASELINE_THRESHOLD = 0.25

    def __init__(self):

        self.settings = get_settings()

        self.retriever = VectorRetriever()

    def collect_candidates(
        self,
        cases: list[EvaluationCase]
    ) -> dict[str, list[RetrievedChunk]]:

        candidates = {}

        max_k = max(self.TOP_K_VALUES)

        for case in cases:

            print(f"Collecting candidates: {case.id}")

            chunks = self.retriever.retrieve(
                question=case.question,
                top_k=max_k,
                min_similarity_score=-1.0
            )

            candidates[case.id] = chunks

        return candidates

    def evaluate_configuration(
        self,
        cases: list[EvaluationCase],
        candidates: dict[str, list[RetrievedChunk]],
        top_k: int,
        threshold: float
    ) -> RetrievalExperimentResult:

        precision_values = []

        recall_values = []

        rr_values = []

        hit_values = []

        context_precision_values = []

        rejection_values = []

        retrieved_counts = []

        full_recall_values = []

        for case in cases:

            chunks = [
                chunk
                for chunk in candidates[case.id][:top_k]
                if chunk.similarity_score >= threshold
            ]

            retrieved_documents = [
                chunk.filename
                for chunk in chunks
            ]

            expected_documents = case.expected_documents

            retrieved_counts.append(
                len(retrieved_documents)
            )

            if not expected_documents:

                rejection_values.append(
                    float(not retrieved_documents)
                )

                continue

            precision = precision_at_k(
                retrieved_documents,
                expected_documents,
                top_k
            )

            recall = recall_at_k(
                retrieved_documents,
                expected_documents,
                top_k
            )

            rr = reciprocal_rank(
                retrieved_documents,
                expected_documents
            )

            hit = hit_at_k(
                retrieved_documents,
                expected_documents,
                top_k
            )

            cp = context_precision(
                retrieved_documents,
                expected_documents
            )

            precision_values.append(precision)

            recall_values.append(recall)

            rr_values.append(rr)

            hit_values.append(float(hit))

            context_precision_values.append(cp)

            full_recall_values.append(
                float(recall == 1.0)
            )

        return RetrievalExperimentResult(
            top_k=top_k,

            min_similarity_score=threshold,

            mean_precision_at_k=mean_metric(
                precision_values
            ),

            mean_recall_at_k=mean_metric(
                recall_values
            ),

            mean_reciprocal_rank=mean_metric(
                rr_values
            ),

            hit_rate=mean_metric(
                hit_values
            ),

            mean_context_precision=mean_metric(
                context_precision_values
            ),

            retrieval_rejection_rate=mean_metric(
                rejection_values
            ),

            mean_retrieved_chunks=(
                sum(retrieved_counts)
                / len(retrieved_counts)
            ),

            full_recall_rate=mean_metric(
                full_recall_values
            )
        )

    @staticmethod
    def select_best_configuration(
        results: list[RetrievalExperimentResult]
    ) -> RetrievalExperimentResult | None:

        # Preserve all expected evidence and reject
        # all queries without expected documents.

        eligible = [
            result
            for result in results
            if result.mean_recall_at_k == 1.0
            and result.retrieval_rejection_rate == 1.0
        ]

        if not eligible:

            return None

        return max(
            eligible,
            key=lambda result: (
                result.mean_context_precision or 0.0,
                result.mean_reciprocal_rank or 0.0,
                -result.mean_retrieved_chunks,
                -result.top_k,
                -result.min_similarity_score
            )
        )

    def run(
        self,
        cases: list[EvaluationCase]
    ) -> RetrievalExperimentReport:

        if not cases:

            raise ValueError(
                "Evaluation dataset is empty"
            )

        indexed_chunks = (
            self.retriever.vector_store.count()
        )

        if indexed_chunks == 0:

            raise ValueError(
                "Vector collection is empty"
            )

        start = perf_counter()

        candidates = self.collect_candidates(cases)

        candidate_latency = (
            perf_counter() - start
        ) * 1000

        results = []

        for top_k, threshold in product(
            self.TOP_K_VALUES,
            self.THRESHOLD_VALUES
        ):

            result = self.evaluate_configuration(
                cases=cases,
                candidates=candidates,
                top_k=top_k,
                threshold=threshold
            )

            results.append(result)

        baseline = next(
            result
            for result in results
            if result.top_k == self.BASELINE_TOP_K
            and result.min_similarity_score
            == self.BASELINE_THRESHOLD
        )

        recommended = self.select_best_configuration(
            results
        )

        return RetrievalExperimentReport(
            evaluated_at=datetime.now(
                timezone.utc
            ).isoformat(),

            embedding_model=(
                self.settings.openai_embedding_model
            ),

            indexed_chunks=indexed_chunks,

            total_cases=len(cases),

            total_configurations=len(results),

            candidate_retrieval_latency_ms=(
                candidate_latency
            ),

            baseline=baseline,

            recommended=recommended,

            results=results
        )