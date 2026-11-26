from app.models.query import QueryRequest

from app.services.rag_service import RAGService

from evaluation.dataset_loader import load_dataset

from evaluation.llm_judge import LLMJudge


def main():

    cases = load_dataset()

    selected_ids = {
        "EVAL-001",
        "EVAL-013",
        "EVAL-014",
        "EVAL-015",
        "EVAL-018"
    }

    service = RAGService()

    judge = LLMJudge()

    for case in cases:

        if case.id not in selected_ids:

            continue

        print("\n" + "=" * 70)

        print(f"CASE: {case.id}")

        print(f"QUESTION: {case.question}")

        trace = service.query_with_trace(
            request=QueryRequest(
                question=case.question,
                top_k=2
            ),

            min_similarity_score=0.30
        )

        print("\nREFERENCE ANSWER")

        print(case.expected_answer)

        print("\nGENERATED ANSWER")

        print(trace.response.answer)

        print("\nRETRIEVED SOURCES")

        for index, chunk in enumerate(
            trace.retrieved_chunks,
            start=1
        ):

            print(
                f"[{index}] {chunk.filename}"
            )

        print("\nJUDGE ASSESSMENT")

        result = judge.evaluate(
            case=case,
            trace=trace
        )

        if result is None:

            print(
                "Semantic evaluation skipped: "
                f"{trace.abstention_reason}"
            )

        else:

            print(
                result.model_dump_json(indent=2)
            )


if __name__ == "__main__":

    main()