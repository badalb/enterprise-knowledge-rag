from time import perf_counter

from app.retrieval.vector_retriever import VectorRetriever
from app.generation.prompt_builder import PromptBuilder
from app.generation.answer_generator import AnswerGenerator
from app.generation.citation_builder import CitationBuilder

from app.models.query import (
    QueryRequest,
    QueryResponse,
    QueryMetrics
)


class RAGService:

    INSUFFICIENT_CONTEXT = (
        "I could not find sufficient information in the "
        "available documents to answer this question."
    )

    def __init__(self):

        self.retriever = VectorRetriever()

        self.prompt_builder = PromptBuilder()

        self.answer_generator = AnswerGenerator()

        self.citation_builder = CitationBuilder()

    def query(
        self,
        request: QueryRequest
    ) -> QueryResponse:

        total_start = perf_counter()

        question = request.question.strip()

        if not question:
            raise ValueError("Question cannot be empty")

        # Step 1: Retrieve relevant chunks
        retrieval_start = perf_counter()

        chunks = self.retriever.retrieve(
            question=question,
            top_k=request.top_k
        )

        retrieval_latency = (
            perf_counter() - retrieval_start
        ) * 1000

        # Step 2: Handle empty retrieval
        if not chunks:

            return QueryResponse(
                question=question,
                answer=self.INSUFFICIENT_CONTEXT,
                sources=[],
                metrics=QueryMetrics(
                    retrieved_chunks=0,
                    retrieval_latency_ms=retrieval_latency,
                    generation_latency_ms=0.0,
                    total_latency_ms=(
                        perf_counter() - total_start
                    ) * 1000,
                    input_tokens=0,
                    output_tokens=0
                )
            )

        # Step 3: Build prompt
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

        # Step 5: Validate and build citations
        answer = result.answer

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

        # Conservative grounding policy
        if invalid_citations or not sources:

            answer = self.INSUFFICIENT_CONTEXT
            sources = []

        # Step 6: Build final response
        return QueryResponse(
            question=question,
            answer=answer,
            sources=sources,
            metrics=QueryMetrics(
                retrieved_chunks=len(chunks),
                retrieval_latency_ms=retrieval_latency,
                generation_latency_ms=generation_latency,
                total_latency_ms=(
                    perf_counter() - total_start
                ) * 1000,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens
            )
        )