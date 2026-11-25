from pathlib import Path

from app.config import get_settings
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.document_chunker import DocumentChunker


def main():

    settings = get_settings()

    loader = DocumentLoader()

    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )

    data_directory = Path("data/sample_documents")

    total_documents = 0
    total_pages = 0
    total_chunks = 0

    for file_path in sorted(data_directory.iterdir()):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in loader.SUPPORTED_EXTENSIONS
        ):
            continue

        document, pages = loader.load(file_path)

        chunks = chunker.chunk(pages)

        total_documents += 1
        total_pages += len(pages)
        total_chunks += len(chunks)

        print("\n" + "=" * 60)

        print(f"Document: {document.metadata.filename}")
        print(f"Document ID: {document.document_id}")
        print(f"File size: {document.metadata.file_size}")
        print(f"Pages: {len(pages)}")
        print(f"Chunks: {len(chunks)}")

        for chunk in chunks:

            print("\n--- Chunk ---")

            print(f"Chunk ID: {chunk.chunk_id}")
            print(f"Index: {chunk.chunk_index}")
            print(f"Page: {chunk.page_number}")
            print(f"Characters: {len(chunk.content)}")
            print(f"Preview: {chunk.content[:150]}")

    print("\n========== INGESTION SUMMARY ==========")

    print(f"Total documents: {total_documents}")
    print(f"Total pages: {total_pages}")
    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()