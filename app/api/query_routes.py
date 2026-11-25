from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from openai import APIError

from app.models.query import QueryRequest, QueryResponse
from app.services.rag_service import RAGService


router = APIRouter(
    prefix="/api/v1",
    tags=["Query"]
)


@lru_cache
def get_rag_service() -> RAGService:

    return RAGService()


@router.post(
    "/query",
    response_model=QueryResponse
)
def query(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service)
):

    try:

        return service.query(request)

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except APIError as exc:

        raise HTTPException(
            status_code=502,
            detail="Upstream AI service request failed"
        ) from exc