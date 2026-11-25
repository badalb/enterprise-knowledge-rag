import re

from app.models.chunk import RetrievedChunk
from app.models.query import SourceCitation


class CitationBuilder:

    CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    def build(
        self,
        answer: str,
        chunks: list[RetrievedChunk]
    ) -> list[SourceCitation]:

        citation_ids = {
            int(match)
            for match in self.CITATION_PATTERN.findall(answer)
        }

        sources = []

        for citation_id in sorted(citation_ids):

            if not 1 <= citation_id <= len(chunks):
                continue

            chunk = chunks[citation_id - 1]

            sources.append(
                SourceCitation(
                    citation_id=citation_id,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    similarity_score=chunk.similarity_score,
                    excerpt=chunk.content[:300]
                )
            )

        return sources

    def has_invalid_citations(
        self,
        answer: str,
        chunk_count: int
    ) -> bool:

        citation_ids = [
            int(match)
            for match in self.CITATION_PATTERN.findall(answer)
        ]

        return any(
            not 1 <= citation_id <= chunk_count
            for citation_id in citation_ids
        )