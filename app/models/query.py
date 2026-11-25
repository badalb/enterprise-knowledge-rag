from pydantic import BaseModel, Field


class QueryRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=4000
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20
    )


class SourceCitation(BaseModel):

    citation_id: int
    chunk_id: str
    document_id: str
    filename: str

    page_number: int | None = None
    similarity_score: float
    excerpt: str


class QueryMetrics(BaseModel):

    retrieved_chunks: int

    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    input_tokens: int | None = None
    output_tokens: int | None = None


class QueryResponse(BaseModel):

    question: str
    answer: str

    sources: list[SourceCitation]
    metrics: QueryMetrics