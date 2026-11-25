from pydantic import BaseModel


class DocumentChunk(BaseModel):

    chunk_id: str
    document_id: str

    filename: str
    page_number: int | None = None
    chunk_index: int

    content: str


class RetrievedChunk(DocumentChunk):

    similarity_score: float