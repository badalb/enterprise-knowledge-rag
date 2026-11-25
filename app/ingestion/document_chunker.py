import hashlib

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from app.models.document import DocumentPage
from app.models.chunk import DocumentChunk


class DocumentChunker:

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def chunk(
        self,
        pages: list[DocumentPage]
    ) -> list[DocumentChunk]:

        chunks = []

        chunk_index = 0

        for page in pages:

            texts = self.splitter.split_text(
                page.content
            )

            for text in texts:

                text = text.strip()

                if not text:
                    continue

                chunk_id = self._generate_chunk_id(
                    page.document_id,
                    chunk_index,
                    text
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=page.document_id,
                        filename=page.filename,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        content=text
                    )
                )

                chunk_index += 1

        return chunks

    @staticmethod
    def _generate_chunk_id(
        document_id: str,
        chunk_index: int,
        content: str
    ) -> str:

        value = (
            f"{document_id}:{chunk_index}:{content}"
        )

        checksum = hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

        return f"CHUNK-{checksum}"