import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.ingestion.models import Chunk, RawDocumentPage
from app.ingestion.loaders import DocumentLoaderRegistry
from app.ingestion.chunker import StructureAwareChunker
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.vector_store import VectorStore, ChromaVectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end ingestion pipeline with resilient file handling, OCR tracking, and deduplication."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or get_embedder()
        self.loader_registry = DocumentLoaderRegistry()

    def run(self, data_dir: Path, reset: bool = False) -> Dict[str, Any]:
        """Execute complete ingestion pipeline for a directory."""
        if not data_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory '{data_dir}' does not exist.")

        if reset:
            logger.info("Reset flag enabled: clearing vector index before ingestion.")
            self.vector_store.reset()

        stats = {
            "files_processed": 0,
            "pages_ocred": 0,
            "chunks_indexed": 0,
            "failures": [],
        }

        # Step 1: Preload structured catalog first if present
        catalog_files = list(data_dir.glob("products.json")) + list(data_dir.glob("products.csv"))
        catalog: Dict[str, Dict[str, Any]] = {}
        for cat_file in catalog_files:
            try:
                loaded = self.loader_registry.load_catalog(cat_file)
                catalog.update(loaded)
                logger.info(f"Loaded {len(loaded)} catalog items from '{cat_file.name}'")
            except Exception as e:
                logger.error(f"Failed to load catalog from '{cat_file.name}': {e}")
                stats["failures"].append({"file": cat_file.name, "error": str(e)})

        chunker = StructureAwareChunker(catalog=catalog)
        all_chunks: List[Chunk] = []

        # Step 2: Iterate all files in data directory
        all_files = [f for f in sorted(data_dir.iterdir()) if f.is_file()]

        for file_path in all_files:
            try:
                logger.info(f"Processing file: {file_path.name}")
                pages = self.loader_registry.load_file(file_path)
                if not pages:
                    logger.warning(f"No pages extracted from '{file_path.name}'")
                    continue

                stats["files_processed"] += 1

                for page in pages:
                    if page.source_type == "ocr" or page.ocr_confidence is not None:
                        stats["pages_ocred"] += 1

                    chunks = chunker.chunk_page(page)
                    all_chunks.extend(chunks)

            except Exception as e:
                logger.exception(f"Error processing file '{file_path.name}': {e}")
                stats["failures"].append({"file": file_path.name, "error": str(e)})

        # Step 3: Embed and index chunks
        if all_chunks:
            # Ensure chunks are deduplicated by chunk_id
            seen_ids = set()
            unique_chunks: List[Chunk] = []
            for c in all_chunks:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    unique_chunks.append(c)
            all_chunks = unique_chunks

            logger.info(f"Computing embeddings for {len(all_chunks)} unique chunks...")
            texts = [c.text for c in all_chunks]
            embeddings = self.embedder.embed_documents(texts)

            logger.info(f"Indexing {len(all_chunks)} chunks into vector store...")
            self.vector_store.add_chunks(all_chunks, embeddings)
            stats["chunks_indexed"] = len(all_chunks)

        logger.info(
            f"Ingestion complete! Summary: files={stats['files_processed']}, "
            f"pages_ocred={stats['pages_ocred']}, chunks={stats['chunks_indexed']}, "
            f"failures={len(stats['failures'])}"
        )

        return stats

