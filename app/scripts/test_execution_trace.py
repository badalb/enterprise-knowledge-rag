from app.models.query import QueryRequest
from app.models.execution_trace import RAGExecutionTrace
from app.services.rag_service import RAGService


def print_trace(
    trace: RAGExecutionTrace
):

    print("\n" + "=" * 70)

    print(f"Question: {trace.question}")

    print("=" * 70)

    print("\nRETRIEVAL CONFIGURATION")

    print(
        trace.retrieval_configuration.model_dump_json(
            indent=2
        )
    )

    print("\nRETRIEVED CONTEXT")

    for index, chunk in enumerate(
        trace.retrieved_chunks,
        start=1
    ):

        print(
            f"[{index}] {chunk.filename} "
            f"(score={chunk.similarity_score:.4f})"
        )

    print("\nGENERATION STATUS")

    print(
        f"Generation attempted: "
        f"{trace.generation_attempted}"
    )

    print(
        f"Abstained: {trace.abstained}"
    )

    print(
        f"Abstention reason: "
        f"{trace.abstention_reason}"
    )

    print(
        f"Invalid citations: "
        f"{trace.invalid_citations}"
    )

    print("\nRAW ANSWER")

    print(trace.raw_answer)

    print("\nFINAL ANSWER")

    print(trace.response.answer)

    print("\nCITED SOURCES")

    for source in trace.response.sources:

        print(
            f"[{source.citation_id}] "
            f"{source.filename}"
        )

    print("\nPERFORMANCE METRICS")

    print(
        trace.response.metrics.model_dump_json(
            indent=2
        )
    )


def main():

    service = RAGService()

    questions = [
        "How do I troubleshoot Kafka consumer lag?",

        "What are the documented RPO and RTO targets?",

        "What authentication and resilience mechanisms "
        "should microservices implement?",

        "What is the current consumer lag of the "
        "production payment consumer group?",

        "Who won the 2011 Cricket World Cup?"
    ]

    for question in questions:

        trace = service.query_with_trace(
            request=QueryRequest(
                question=question,
                top_k=2
            ),

            min_similarity_score=0.30
        )

        print_trace(trace)


if __name__ == "__main__":

    main()