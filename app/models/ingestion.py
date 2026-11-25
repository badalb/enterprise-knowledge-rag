from pydantic import BaseModel


class IngestionResponse(BaseModel):

    document_id: str
    filename: str
    status: str
    chunks: int
    reason: str | None = None


class IndexedDocument(BaseModel):

    document_id: str
    filename: str
    chunk_count: int


class DeleteDocumentResponse(BaseModel):

    document_id: str
    status: str