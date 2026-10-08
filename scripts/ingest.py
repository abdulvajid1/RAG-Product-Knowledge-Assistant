import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import setup_logging
from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger("scripts.ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingest product knowledge base into vector store.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to raw knowledge base directory (default: data/raw)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset/clear the vector index before ingesting",
    )
    args = parser.parse_args()

    setup_logging()
    data_path = Path(args.data_dir).resolve()

    logger.info(f"Starting ingestion from: {data_path}")
    logger.info(f"Reset index: {args.reset}")

    pipeline = IngestionPipeline()
    stats = pipeline.run(data_dir=data_path, reset=args.reset)

    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Files Processed: {stats['files_processed']}")
    print(f"Pages OCR'd:     {stats['pages_ocred']}")
    print(f"Chunks Indexed:  {stats['chunks_indexed']}")
    print(f"Failures:        {len(stats['failures'])}")
    if stats["failures"]:
        print("\nFailures Detail:")
        for fail in stats["failures"]:
            print(f" - {fail['file']}: {fail['error']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

