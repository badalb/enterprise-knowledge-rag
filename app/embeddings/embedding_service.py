from openai import OpenAI

from app.config import get_settings


class EmbeddingService:

    def __init__(self):

        settings = get_settings()

        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

        self.model = settings.openai_embedding_model

    def embed_text(
        self,
        text: str
    ) -> list[float]:

        return self.embed_texts([text])[0]

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 100
    ) -> list[list[float]]:

        if not texts:
            return []

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if any(not text.strip() for text in texts):
            raise ValueError("Empty text cannot be embedded")

        embeddings = []

        for start in range(0, len(texts), batch_size):

            batch = texts[start:start + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )

            ordered_data = sorted(
                response.data,
                key=lambda item: item.index
            )

            embeddings.extend(
                item.embedding
                for item in ordered_data
            )

        return embeddings