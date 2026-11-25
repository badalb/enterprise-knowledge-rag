from app.embeddings.embedding_service import EmbeddingService


def main():

    service = EmbeddingService()

    texts = [
        "Kafka consumer lag is increasing.",
        "Spark jobs are experiencing data skew.",
        "Database connection pool is exhausted."
    ]

    embeddings = service.embed_texts(texts)

    print(f"Total embeddings: {len(embeddings)}")

    for text, embedding in zip(texts, embeddings):

        print(f"\nText: {text}")
        print(f"Dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")


if __name__ == "__main__":
    main()