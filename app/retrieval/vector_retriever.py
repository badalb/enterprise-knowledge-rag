from app.config import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_store import ChromaStore
from app.models.chunk import RetrievedChunk


class VectorRetriever:

    def __init__(self):

        self.settings = get_settings()

        self.embedding_service = EmbeddingService()

        self.vector_store = ChromaStore()

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        min_similarity_score: float | None = None
    ) -> list[RetrievedChunk]:

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty")

        if top_k is None:
            top_k = self.settings.top_k

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if min_similarity_score is None:
            min_similarity_score = (
                self.settings.min_similarity_score
            )

        if not -1.0 <= min_similarity_score <= 1.0:
            raise ValueError(
                "Similarity threshold must be between -1 and 1"
            )

        if self.vector_store.count() == 0:
            return []

        # Generate query embedding
        query_embedding = (
            self.embedding_service.embed_text(question)
        )

        # Search vector database
        result = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        retrieved_chunks = []

        for chunk_id, content, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0]
        ):

            similarity_score = max(
                -1.0,
                min(1.0, 1.0 - float(distance))
            )

            if similarity_score < min_similarity_score:
                continue

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=metadata["document_id"],
                    filename=metadata["filename"],
                    page_number=metadata.get("page_number"),
                    chunk_index=metadata["chunk_index"],
                    content=content,
                    similarity_score=similarity_score
                )
            )

        return sorted(
            retrieved_chunks,
            key=lambda chunk: chunk.similarity_score,
            reverse=True
        )