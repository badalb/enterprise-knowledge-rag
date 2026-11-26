from time import perf_counter

from app.config import get_settings

from app.generation.citation_parser import CitationParser

from app.retrieval.vector_retriever import (
    VectorRetriever
)

from app.generation.prompt_builder import (
    PromptBuilder
)

from app.generation.answer_generator import (
    AnswerGenerator
)

from app.generation.citation_builder import (
    CitationBuilder
)

from app.models.query import (
    QueryRequest,
    QueryResponse,
    QueryMetrics
)

from app.models.execution_trace import (
    RAGExecutionTrace,
    RetrievalConfiguration,
    AbstentionReason
)


class RAGService:

    INSUFFICIENT_CONTEXT = (
        "I could not find sufficient information "
        "in the available documents to answer this question."
    )

    def __init__(self):

        self.settings = get_settings()

        self.retriever = VectorRetriever()

        self.prompt_builder = PromptBuilder()

        self.answer_generator = AnswerGenerator()

        self.citation_builder = CitationBuilder()

    def query(
        self,
        request: QueryRequest
    ) -> QueryResponse:

        trace = self.query_with_trace(request)

        return trace.response

    def query_with_trace(
        self,
        request: QueryRequest,
        min_similarity_score: float | None = None
    ) -> RAGExecutionTrace:

        total_start = perf_counter()

        question = request.question.strip()

        if not question:

            raise ValueError(
                "Question cannot be empty"
            )

        # Resolve effective retrieval configuration

        top_k = (
            request.top_k
            if request.top_k is not None
            else self.settings.top_k
        )

        threshold = (
            min_similarity_score
            if min_similarity_score is not None
            else self.settings.min_similarity_score
        )

        configuration = RetrievalConfiguration(
            top_k=top_k,
            min_similarity_score=threshold
        )

        # Step 1: Retrieve relevant context

        retrieval_start = perf_counter()

        chunks = self.retriever.retrieve(
            question=question,
            top_k=configuration.top_k,
            min_similarity_score=(
                configuration.min_similarity_score
            )
        )

        retrieval_latency = (
            perf_counter() - retrieval_start
        ) * 1000

        # Step 2: Handle empty retrieval

        if not chunks:

            response = QueryResponse(
                question=question,

                answer=self.INSUFFICIENT_CONTEXT,

                sources=[],

                metrics=QueryMetrics(
                    retrieved_chunks=0,

                    retrieval_latency_ms=(
                        retrieval_latency
                    ),

                    generation_latency_ms=0.0,

                    total_latency_ms=(
                        perf_counter() - total_start
                    ) * 1000,

                    input_tokens=0,

                    output_tokens=0
                )
            )

            return RAGExecutionTrace(
                question=question,

                retrieval_configuration=configuration,

                retrieved_chunks=[],

                generation_attempted=False,

                raw_answer=None,

                invalid_citations=False,

                abstained=True,

                abstention_reason=(
                    "NO_RETRIEVED_CONTEXT"
                ),

                response=response
            )

        # Step 3: Construct the generation prompt

        messages = self.prompt_builder.build(
            question=question,
            chunks=chunks
        )

        # Step 4: Generate answer

        generation_start = perf_counter()

        result = self.answer_generator.generate(
            messages
        )

        generation_latency = (
            perf_counter() - generation_start
        ) * 1000

        raw_answer = result.answer

        answer = CitationParser.normalize(raw_answer)

        # Step 5: Build and validate citations

        sources = self.citation_builder.build(
            answer=answer,
            chunks=chunks
        )

        invalid_citations = (
            self.citation_builder.has_invalid_citations(
                answer=answer,
                chunk_count=len(chunks)
            )
        )

        # Step 6: Determine abstention reason

        abstained = False

        abstention_reason: AbstentionReason | None = None

        if invalid_citations:

            abstained = True

            abstention_reason = "INVALID_CITATIONS"

        elif raw_answer.strip() == self.INSUFFICIENT_CONTEXT:

            abstained = True

            abstention_reason = "MODEL_ABSTENTION"

        elif not sources:

            abstained = True

            abstention_reason = "MISSING_CITATIONS"

        # Step 7: Apply application guardrails

        if abstained:

            answer = self.INSUFFICIENT_CONTEXT

            sources = []

        # Step 8: Construct public response

        response = QueryResponse(
            question=question,

            answer=answer,

            sources=sources,

            metrics=QueryMetrics(
                retrieved_chunks=len(chunks),

                retrieval_latency_ms=(
                    retrieval_latency
                ),

                generation_latency_ms=(
                    generation_latency
                ),

                total_latency_ms=(
                    perf_counter() - total_start
                ) * 1000,

                input_tokens=result.input_tokens,

                output_tokens=result.output_tokens
            )
        )

        # Step 9: Construct internal execution trace

        return RAGExecutionTrace(
            question=question,

            retrieval_configuration=configuration,

            retrieved_chunks=chunks,

            generation_attempted=True,

            raw_answer=raw_answer,

            invalid_citations=invalid_citations,

            abstained=abstained,

            abstention_reason=abstention_reason,

            response=response
        )