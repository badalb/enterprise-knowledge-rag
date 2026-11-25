from time import perf_counter

from app.retrieval.vector_retriever import VectorRetriever


def main():

    retriever = VectorRetriever()

    questions = [
        "How do I troubleshoot Kafka consumer lag?",
        "Why are Spark tasks taking a long time?",
        "What is the disaster recovery RPO for payments?",
        "How should database connection pools be monitored?",
        "How are internal APIs authenticated?",
        "Who won the 2011 Cricket World Cup?"
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print(f"Question: {question}")

        start = perf_counter()

        chunks = retriever.retrieve(
            question=question,
            top_k=5
        )

        elapsed_ms = (
            perf_counter() - start
        ) * 1000

        print(f"Retrieved chunks: {len(chunks)}")
        print(f"Retrieval latency: {elapsed_ms:.2f} ms")

        if not chunks:
            print("No relevant documents found.")
            continue

        for rank, chunk in enumerate(chunks, start=1):

            print(f"\n--- Rank {rank} ---")

            print(f"Document: {chunk.filename}")

            print(
                f"Similarity: "
                f"{chunk.similarity_score:.4f}"
            )

            print(f"Page: {chunk.page_number}")

            print(f"Preview: {chunk.content[:200]}")


if __name__ == "__main__":
    main()