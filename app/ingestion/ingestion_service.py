from pathlib import Path

from app.config import get_settings
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.document_chunker import DocumentChunker
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore


class IngestionService:

    def __init__(self):

        settings = get_settings()

        self.loader = DocumentLoader()

        self.chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        self.embedding_service = EmbeddingService()

        self.vector_store = ChromaStore()

    def ingest(
        self,
        file_path: str | Path
    ) -> dict:

        document, pages = self.loader.load(file_path)

        document_id = document.document_id

        if self.vector_store.document_exists(document_id):

            return {
                "document_id": document_id,
                "filename": document.metadata.filename,
                "status": "SKIPPED",
                "reason": "Document already indexed",
                "chunks": 0
            }

        chunks = self.chunker.chunk(pages)

        if not chunks:
            raise ValueError("No chunks generated")

        texts = [
            chunk.content
            for chunk in chunks
        ]

        embeddings = self.embedding_service.embed_texts(
            texts
        )

        self.vector_store.add_chunks(
            chunks,
            embeddings
        )

        return {
            "document_id": document_id,
            "filename": document.metadata.filename,
            "status": "INDEXED",
            "chunks": len(chunks)
        }