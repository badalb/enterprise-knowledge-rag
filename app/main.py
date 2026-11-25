from fastapi import FastAPI

from app.config import get_settings

from app.api.query_routes import router as query_router
from app.api.document_routes import router as document_router


settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Enterprise Document Question Answering "
        "using Naive RAG"
    )
)


app.include_router(query_router)

app.include_router(document_router)


@app.get(
    "/api/v1/health",
    tags=["Health"]
)
def health_check():

    return {
        "status": "UP",
        "application": settings.app_name,
        "version": settings.app_version
    }