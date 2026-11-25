import hashlib
from pathlib import Path

from pypdf import PdfReader

from app.models.document import (
    Document,
    DocumentMetadata,
    DocumentPage
)


class DocumentLoader:

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def load(
        self,
        file_path: str | Path
    ) -> tuple[Document, list[DocumentPage]]:

        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        file_bytes = path.read_bytes()

        checksum = hashlib.sha256(
            file_bytes
        ).hexdigest()

        document_id = f"DOC-{checksum}"

        metadata = DocumentMetadata(
            filename=path.name,
            file_type=extension.lstrip("."),
            file_size=len(file_bytes),
            checksum=checksum
        )

        document = Document(
            document_id=document_id,
            metadata=metadata
        )

        if extension == ".pdf":
            pages = self._load_pdf(path, document_id)
        else:
            pages = self._load_text(path, document_id)

        if not pages:
            raise ValueError(
                f"No extractable content found: {path.name}"
            )

        return document, pages

    def _load_pdf(
        self,
        path: Path,
        document_id: str
    ) -> list[DocumentPage]:

        reader = PdfReader(path)

        pages = []

        for page_number, page in enumerate(
            reader.pages, start=1
        ):

            content = (page.extract_text() or "").strip()

            if not content:
                continue

            pages.append(
                DocumentPage(
                    document_id=document_id,
                    filename=path.name,
                    page_number=page_number,
                    content=content
                )
            )

        return pages

    def _load_text(
        self,
        path: Path,
        document_id: str
    ) -> list[DocumentPage]:

        content = path.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            return []

        return [
            DocumentPage(
                document_id=document_id,
                filename=path.name,
                page_number=None,
                content=content
            )
        ]