from pathlib import Path

from evaluation.dataset_loader import load_dataset


DOCUMENT_DIRECTORY = Path(
    "data/sample_documents"
)

OUTPUT_FILE = Path(
    "evaluation/datasets/reference_answer_review.md"
)


def main():

    cases = load_dataset()

    lines = [
        "# Reference Answer Review",
        "",
        "Review each question against its source documents.",
        "",
        "Define the expected answer and essential facts.",
        ""
    ]

    for case in cases:

        lines.extend([
            "---",
            "",
            f"## {case.id}",
            "",
            f"**Question:** {case.question}",
            "",
            f"**Category:** {case.category}",
            "",
            f"**Answerable:** {case.answerable}",
            ""
        ])

        for filename in case.expected_documents:

            path = DOCUMENT_DIRECTORY / filename

            if not path.is_file():

                raise FileNotFoundError(
                    f"Missing source document: {path}"
                )

            content = path.read_text(
                encoding="utf-8"
            )

            lines.extend([
                f"### Source: {filename}",
                "",
                content,
                ""
            ])

        lines.extend([
            "### Expected Answer",
            "",
            case.expected_answer or "",
            "",
            "### Expected Facts",
            ""
        ])

        for fact in case.expected_facts:

            lines.append(f"- {fact}")

        lines.append("")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(
        f"Reference review generated: {OUTPUT_FILE}"
    )

    print(f"Total evaluation cases: {len(cases)}")


if __name__ == "__main__":

    main()