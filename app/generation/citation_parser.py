import re


class CitationParser:

    CITATION_PATTERN = re.compile(
        r"\[\s*"
        r"((?:SOURCE\s+)?\d+"
        r"(?:\s*,\s*(?:SOURCE\s+)?\d+)*)"
        r"\s*\]",
        flags=re.IGNORECASE
    )

    CANONICAL_PATTERN = re.compile(
        r"\[(\d+)\]"
    )

    @classmethod
    def normalize(cls, answer: str) -> str:

        def replace_citation(match):

            citation_ids = re.findall(
                r"\d+",
                match.group(1)
            )

            return "".join(
                f"[{int(citation_id)}]"
                for citation_id in citation_ids
            )

        return cls.CITATION_PATTERN.sub(
            replace_citation,
            answer
        )

    @classmethod
    def extract_ids(cls, answer: str) -> list[int]:

        normalized_answer = cls.normalize(answer)

        return [
            int(citation_id)
            for citation_id in
            cls.CANONICAL_PATTERN.findall(
                normalized_answer
            )
        ]

    @classmethod
    def has_invalid_citations(
        cls,
        answer: str,
        chunk_count: int
    ) -> bool:

        return any(
            not 1 <= citation_id <= chunk_count
            for citation_id in cls.extract_ids(answer)
        )