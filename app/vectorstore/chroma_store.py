import chromadb

from app.config import get_settings
from app.models.chunk import DocumentChunk


class ChromaStore:

    def __init__(self):

        settings = get_settings()

        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            },
            embedding_function=None
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]]
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunk and embedding counts must match"
            )

        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            ids.append(chunk.chunk_id)
            documents.append(chunk.content)

            metadata = {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index
            }

            if chunk.page_number is not None:
                metadata["page_number"] = chunk.page_number

            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def document_exists(
        self,
        document_id: str
    ) -> bool:

        result = self.collection.get(
            where={
                "document_id": document_id
            },
            limit=1,
            include=[]
        )

        return bool(result["ids"])

    def delete_document(
        self,
        document_id: str
    ) -> None:

        self.collection.delete(
            where={
                "document_id": document_id
            }
        )

    def count(self) -> int:

        return self.collection.count()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> dict:

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        total_chunks = self.collection.count()

        if total_chunks == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_chunks),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    def list_documents(self) -> list[dict]:

        result = self.collection.get(
            include=["metadatas"]
        )

        documents = {}

        for metadata in result["metadatas"] or []:

            document_id = metadata["document_id"]

            if document_id not in documents:

                documents[document_id] = {
                    "document_id": document_id,
                    "filename": metadata["filename"],
                    "chunk_count": 0
                }

            documents[document_id]["chunk_count"] += 1

        return sorted(
            documents.values(),
            key=lambda document: document["filename"]
        )