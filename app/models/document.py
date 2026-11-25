from datetime import datetime, timezone

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):

    filename: str
    file_type: str
    file_size: int
    checksum: str

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Document(BaseModel):

    document_id: str
    metadata: DocumentMetadata


class DocumentPage(BaseModel):

    document_id: str
    filename: str
    page_number: int | None = None
    content: str