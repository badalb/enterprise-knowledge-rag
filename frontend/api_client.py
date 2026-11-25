import os

import httpx


class APIClient:

    def __init__(self):

        self.base_url = os.getenv(
            "RAG_API_URL",
            "http://localhost:8000"
        ).rstrip("/")

        self.timeout = 120.0

    def health(self) -> dict:

        response = httpx.get(
            f"{self.base_url}/api/v1/health",
            timeout=5.0
        )

        response.raise_for_status()

        return response.json()

    def query(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:

        response = httpx.post(
            f"{self.base_url}/api/v1/query",
            json={
                "question": question,
                "top_k": top_k
            },
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()

    def list_documents(self) -> list[dict]:

        response = httpx.get(
            f"{self.base_url}/api/v1/documents",
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()

    def upload_document(
        self,
        filename: str,
        content: bytes,
        content_type: str
    ) -> dict:

        response = httpx.post(
            f"{self.base_url}/api/v1/documents",
            files={
                "file": (
                    filename,
                    content,
                    content_type
                )
            },
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()

    def delete_document(
        self,
        document_id: str
    ) -> dict:

        response = httpx.delete(
            f"{self.base_url}/api/v1/documents/{document_id}",
            timeout=self.timeout
        )

        response.raise_for_status()

        return response.json()