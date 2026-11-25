import logging

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File
)

from openai import APIError

from app.ingestion.ingestion_service import IngestionService

from app.models.ingestion import (
    IngestionResponse,
    IndexedDocument,
    DeleteDocumentResponse
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"]
)


MAX_FILE_SIZE = 10 * 1024 * 1024


@lru_cache
def get_ingestion_service() -> IngestionService:

    return IngestionService()


@router.post(
    "",
    response_model=IngestionResponse
)
def upload_document(
    file: UploadFile = File(...),
    service: IngestionService = Depends(
        get_ingestion_service
    )
):

    filename = Path(file.filename or "").name

    if not filename or filename in {".", ".."}:

        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    extension = Path(filename).suffix.lower()

    if extension not in service.loader.SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {extension}"
        )

    try:

        with TemporaryDirectory() as directory:

            file_path = Path(directory) / filename

            total_size = 0

            with file_path.open("wb") as destination:

                while True:

                    content = file.file.read(1024 * 1024)

                    if not content:
                        break

                    total_size += len(content)

                    if total_size > MAX_FILE_SIZE:

                        raise HTTPException(
                            status_code=413,
                            detail="File exceeds 10 MB limit"
                        )

                    destination.write(content)

            if total_size == 0:

                raise HTTPException(
                    status_code=400,
                    detail="Empty file"
                )

            return service.ingest(file_path)

    except HTTPException:

        raise

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except APIError as exc:

        raise HTTPException(
            status_code=502,
            detail="Embedding service request failed"
        ) from exc

    except Exception as exc:

        logger.exception("Document ingestion failed")

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed"
        ) from exc

    finally:

        file.file.close()


@router.get(
    "",
    response_model=list[IndexedDocument]
)
def list_documents(
    service: IngestionService = Depends(
        get_ingestion_service
    )
):

    return service.vector_store.list_documents()


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse
)
def delete_document(
    document_id: str,
    service: IngestionService = Depends(
        get_ingestion_service
    )
):

    if not service.vector_store.document_exists(
        document_id
    ):

        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    service.vector_store.delete_document(
        document_id
    )

    return DeleteDocumentResponse(
        document_id=document_id,
        status="DELETED"
    )