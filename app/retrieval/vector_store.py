import abc
import os
import shutil
import logging
from typing import List, Tuple, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.ingestion.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


class VectorStore(abc.ABC):
    """Abstract interface for vector index operations."""

    @abc.abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        pass

    @abc.abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Return list of (Chunk, similarity_score) sorted by relevance descending."""
        pass

    @abc.abstractmethod
    def count(self) -> int:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB persistent vector store implementation."""

    COLLECTION_NAME = "filumart_products"

    def __init__(self, persist_directory: Optional[str] = None):
        settings = get_settings()
        self.persist_dir = persist_directory or settings.vector_store_path
        os.makedirs(self.persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Clear all indexed chunks."""
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Chroma vector store collection reset.")

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        if not chunks:
            return

        # Deduplicate incoming chunks by chunk_id to satisfy ChromaDB unique ID requirement
        seen_ids = set()
        unique_chunks: List[Chunk] = []
        unique_embeddings: Optional[List[List[float]]] = [] if embeddings is not None else None

        for idx, c in enumerate(chunks):
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                unique_chunks.append(c)
                if embeddings is not None:
                    unique_embeddings.append(embeddings[idx])

        chunks = unique_chunks
        embeddings = unique_embeddings

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []

        for c in chunks:
            # Flatten metadata for Chroma compatibility (only str, int, float, bool)
            meta_dict = {}
            for k, v in c.metadata.model_dump().items():
                if v is not None:
                    meta_dict[k] = v
                else:
                    meta_dict[k] = ""  # Chroma doesn't accept None in metadata values
            metadatas.append(meta_dict)

        kwargs: Dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        self._collection.upsert(**kwargs)
        logger.info(f"Upserted {len(chunks)} chunks into Chroma collection.")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        if self._collection.count() == 0:
            return []

        where_filter = self._build_where_filter(filters)

        query_args: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count()),
        }
        if where_filter:
            query_args["where"] = where_filter

        try:
            results = self._collection.query(**query_args)
        except Exception as e:
            logger.error(f"Error querying Chroma vector store: {e}")
            return []

        chunks_with_scores: List[Tuple[Chunk, float]] = []

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            c_id = ids[i]
            c_text = docs[i]
            meta_dict = metas[i] if i < len(metas) else {}
            dist = distances[i] if i < len(distances) else 1.0

            # Cosine similarity for Chroma cosine space: sim = 1.0 - distance
            similarity = max(0.0, min(1.0, 1.0 - dist))

            # Reconstruct ChunkMetadata
            chunk_meta = ChunkMetadata(
                product_id=meta_dict.get("product_id") or None,
                product_name=meta_dict.get("product_name") or None,
                category=meta_dict.get("category") or None,
                brand=meta_dict.get("brand") or None,
                supplier=meta_dict.get("supplier") or None,
                country=meta_dict.get("country") or None,
                document=meta_dict.get("document", ""),
                page=int(meta_dict.get("page", 1)),
                section=meta_dict.get("section") or None,
                source_type=meta_dict.get("source_type", "text"),
                ocr_confidence=float(meta_dict["ocr_confidence"]) if meta_dict.get("ocr_confidence") != "" and meta_dict.get("ocr_confidence") is not None else None,
                low_confidence=bool(meta_dict.get("low_confidence", False)),
            )

            chunk = Chunk(chunk_id=c_id, text=c_text, metadata=chunk_meta)
            chunks_with_scores.append((chunk, round(similarity, 4)))

        # Sort by similarity descending
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        return chunks_with_scores

    def _build_where_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Convert flat filter dict to Chroma $and filter."""
        if not filters:
            return None

        clauses = []
        for key in ["category", "brand", "supplier", "country", "product_id"]:
            val = filters.get(key)
            if val is not None and str(val).strip():
                clauses.append({key: {"$eq": str(val).strip()}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

