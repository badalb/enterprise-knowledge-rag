from typing import Literal

from pydantic import BaseModel, Field

from app.models.chunk import RetrievedChunk
from app.models.query import QueryResponse


AbstentionReason = Literal[
    "NO_RETRIEVED_CONTEXT",
    "MODEL_ABSTENTION",
    "INVALID_CITATIONS",
    "MISSING_CITATIONS"
]


class RetrievalConfiguration(BaseModel):

    top_k: int = Field(
        ge=1,
        le=20
    )

    min_similarity_score: float = Field(
        ge=-1.0,
        le=1.0
    )


class RAGExecutionTrace(BaseModel):

    question: str

    retrieval_configuration: RetrievalConfiguration

    retrieved_chunks: list[RetrievedChunk]

    generation_attempted: bool

    raw_answer: str | None = None

    invalid_citations: bool = False

    abstained: bool = False

    abstention_reason: AbstentionReason | None = None

    response: QueryResponse