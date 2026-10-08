import abc
import hashlib
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.config import get_settings


class Embedder(abc.ABC):
    """Abstract interface for text embedding models."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        pass

    @abc.abstractmethod
    def embed_query(self, query: str) -> List[float]:
        pass

    @abc.abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        pass


class SentenceTransformerEmbedder(Embedder):
    """Embedding model based on SentenceTransformers (e.g. BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self.model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, query: str) -> List[float]:
        embedding = self._model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        if not documents:
            return []
        embeddings = self._model.encode(documents, normalize_embeddings=True)
        return embeddings.tolist()


class MockEmbedder(Embedder):
    """Deterministic hash-based embedder for offline/unit testing without downloading weights."""

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_vector(self, text: str) -> List[float]:
        vec = []
        for i in range(self._dim):
            h = hashlib.sha256(f"{text}_{i}".encode()).digest()
            val = (int.from_bytes(h[:4], "little") / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self._hash_vector(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self._hash_vector(doc) for doc in documents]


_cached_embedder: Optional[Embedder] = None


def get_embedder(use_mock: bool = False) -> Embedder:
    """Return singleton embedder instance."""
    global _cached_embedder
    if _cached_embedder is not None and not use_mock:
        return _cached_embedder

    if use_mock:
        return MockEmbedder()

    try:
        _cached_embedder = SentenceTransformerEmbedder()
        return _cached_embedder
    except Exception:
        # Fallback to mock embedder if weights cannot be loaded
        return MockEmbedder()

