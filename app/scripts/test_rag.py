from app.services.rag_service import RAGService
from app.models.query import QueryRequest


def main():

    service = RAGService()

    questions = [
        "How do I troubleshoot Kafka consumer lag?",

        "How can I identify and resolve Spark data skew?",

        "What is the RPO and RTO for the payment platform?",

        "How should database connection pools be monitored?",

        "How are internal APIs authenticated?",

        "Who won the 2011 Cricket World Cup?"
    ]

    for question in questions:

        print("\n" + "=" * 70)

        request = QueryRequest(
            question=question,
            top_k=5
        )

        response = service.query(request)

        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()