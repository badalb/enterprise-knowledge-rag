from pathlib import Path
from time import perf_counter

from app.ingestion.ingestion_service import IngestionService


def main():

    service = IngestionService()

    data_directory = Path("data/sample_documents")

    indexed = 0
    skipped = 0
    failed = 0
    total_chunks = 0

    start = perf_counter()

    for file_path in sorted(data_directory.iterdir()):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in service.loader.SUPPORTED_EXTENSIONS
        ):
            continue

        try:

            result = service.ingest(file_path)

            print(
                f"{result['status']}: "
                f"{result['filename']} "
                f"(chunks={result['chunks']})"
            )

            if result["status"] == "INDEXED":
                indexed += 1
                total_chunks += result["chunks"]
            else:
                skipped += 1

        except Exception as exc:

            failed += 1

            print(
                f"FAILED: {file_path.name}: {exc}"
            )

    elapsed = perf_counter() - start

    print("\n========== INGESTION SUMMARY ==========")

    print(f"Indexed documents: {indexed}")
    print(f"Skipped documents: {skipped}")
    print(f"Failed documents: {failed}")
    print(f"New chunks: {total_chunks}")

    print(
        f"Total stored chunks: "
        f"{service.vector_store.count()}"
    )

    print(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()