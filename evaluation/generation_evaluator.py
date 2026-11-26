from collections import defaultdict

from datetime import datetime, timezone

from time import perf_counter

from app.config import get_settings

from app.models.query import QueryRequest

from app.services.rag_service import RAGService

from evaluation.llm_judge import LLMJudge

from evaluation.models import (
    EvaluationCase,
    GenerationEvaluationResult,
    GenerationEvaluationSummary,
    GenerationEvaluationReport
)

from evaluation.generation_metrics import (
    evaluate_deterministic,
    summarize_deterministic
)

from evaluation.semantic_metrics import (
    summarize_semantic
)


class GenerationEvaluator:

    def __init__(
        self,
        service: RAGService | None = None,
        judge: LLMJudge | None = None
    ):

        self.settings = get_settings()

        self.service = (
            service
            if service is not None
            else RAGService()
        )

        self.judge = (
            judge
            if judge is not None
            else LLMJudge()
        )

    def evaluate_case(
        self,
        case: EvaluationCase,
        top_k: int,
        min_similarity_score: float
    ) -> GenerationEvaluationResult:

        # Step 1: Execute the actual RAG pipeline

        trace = self.service.query_with_trace(
            request=QueryRequest(
                question=case.question,
                top_k=top_k
            ),

            min_similarity_score=(
                min_similarity_score
            )
        )

        # Step 2: Deterministic evaluation

        deterministic = evaluate_deterministic(
            case=case,
            trace=trace
        )

        # Step 3: Semantic evaluation

        semantic = self.judge.evaluate(
            case=case,
            trace=trace
        )

        # Step 4: Combine evaluation results

        return GenerationEvaluationResult(
            case=case,

            trace=trace,

            deterministic=deterministic,

            semantic=semantic
        )

    def evaluate(
        self,
        cases: list[EvaluationCase],
        top_k: int | None = None,
        min_similarity_score: float | None = None
    ) -> GenerationEvaluationReport:

        if not cases:

            raise ValueError(
                "Evaluation dataset is empty"
            )

        if top_k is None:

            top_k = self.settings.top_k

        if min_similarity_score is None:

            min_similarity_score = (
                self.settings.min_similarity_score
            )

        if not 1 <= top_k <= 20:

            raise ValueError("Invalid top_k")

        if not -1.0 <= min_similarity_score <= 1.0:

            raise ValueError(
                "Invalid similarity threshold"
            )

        indexed_chunks = (
            self.service.retriever.vector_store.count()
        )

        if indexed_chunks == 0:

            raise ValueError(
                "Vector collection is empty. "
                "Ingest documents before evaluation."
            )

        # Validate reference answers before API calls

        for case in cases:

            if not case.expected_answer:

                raise ValueError(
                    f"Missing reference answer: {case.id}"
                )

            if case.answerable and not case.expected_facts:

                raise ValueError(
                    f"Missing expected facts: {case.id}"
                )

        results = []

        start = perf_counter()

        for index, case in enumerate(
            cases,
            start=1
        ):

            print(
                f"\n[{index}/{len(cases)}] "
                f"{case.id}: {case.question}"
            )

            result = self.evaluate_case(
                case=case,

                top_k=top_k,

                min_similarity_score=(
                    min_similarity_score
                )
            )

            results.append(result)

            self.print_case_status(result)

        evaluation_duration_ms = (
            perf_counter() - start
        ) * 1000

        grouped_results = defaultdict(list)

        for result in results:

            grouped_results[
                result.case.category
            ].append(result)

        category_summaries = {
            category: self.summarize(category_results)

            for category, category_results
            in grouped_results.items()
        }

        return GenerationEvaluationReport(
            evaluated_at=datetime.now(
                timezone.utc
            ).isoformat(),

            embedding_model=(
                self.settings.openai_embedding_model
            ),

            generation_model=(
                self.settings.openai_chat_model
            ),

            judge_model=self.judge.model,

            collection_name=(
                self.settings.chroma_collection_name
            ),

            indexed_chunks=indexed_chunks,

            top_k=top_k,

            min_similarity_score=(
                min_similarity_score
            ),

            evaluation_duration_ms=(
                evaluation_duration_ms
            ),

            summary=self.summarize(results),

            category_summaries=category_summaries,

            results=results
        )

    @staticmethod
    def summarize(
        results: list[GenerationEvaluationResult]
    ) -> GenerationEvaluationSummary:

        deterministic = summarize_deterministic([
            result.deterministic
            for result in results
        ])

        semantic = summarize_semantic(results)

        return GenerationEvaluationSummary(
            deterministic=deterministic,

            semantic=semantic
        )

    @staticmethod
    def print_case_status(
        result: GenerationEvaluationResult
    ):

        deterministic = result.deterministic

        semantic = result.semantic

        print(
            f"  Retrieved chunks: "
            f"{deterministic.retrieved_chunks}"
        )

        print(
            f"  Abstention correct: "
            f"{deterministic.abstention_correct}"
        )

        if result.trace.abstained:

            print(
                f"  Abstention reason: "
                f"{result.trace.abstention_reason}"
            )

        if semantic is not None:

            print(
                f"  Correctness: "
                f"{semantic.correctness:.4f}"
            )

            print(
                f"  Faithfulness: "
                f"{semantic.faithfulness:.4f}"
            )

            print(
                f"  Citation support: "
                f"{semantic.citation_support:.4f}"
            )

            if semantic.fact_coverage is not None:

                print(
                    f"  Fact coverage: "
                    f"{semantic.fact_coverage:.4f}"
                )

        else:

            print("  Semantic evaluation skipped")