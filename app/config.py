from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_name: str = "Enterprise Knowledge RAG"
    app_version: str = "1.0.0"

    # OpenAI
    openai_api_key: str = Field(min_length=1)
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Chunking
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    # Retrieval
    top_k: int = Field(default=5, ge=1, le=20)

    min_similarity_score: float = Field(
        default=0.25,
        ge=-1.0,
        le=1.0
)

    # ChromaDB
    chroma_persist_directory: str = "./data/chroma"
    chroma_collection_name: str = "enterprise_documents"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode="after")
    def validate_chunking(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
