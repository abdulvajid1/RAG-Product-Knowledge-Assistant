import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from app.ingestion.models import Chunk

logger = logging.getLogger(__name__)


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens and product codes."""
    if not text:
        return []
    # Retain alphanumeric tokens, product IDs (e.g. P001), model numbers (SolarMax, 550W, 85°C)
    tokens = re.findall(r'[a-zA-Z0-9°]+', text.lower())
    return tokens


class BM25Searcher:
    """In-memory BM25 index for exact token and model number matching."""

    def __init__(self, chunks: Optional[List[Chunk]] = None):
        self.chunks: List[Chunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build BM25 index from a list of chunks."""
        self.chunks = chunks
        self.corpus_tokens = [tokenize_for_bm25(c.text) for c in chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            logger.info(f"Built BM25 index with {len(chunks)} chunks.")
        else:
            self.bm25 = None

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Rank chunks using BM25 with optional metadata pre-filtering.
        Returns list of (Chunk, normalized_score) descending.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0

        scored_chunks: List[Tuple[Chunk, float]] = []

        for idx, chunk in enumerate(self.chunks):
            raw_s = raw_scores[idx]
            if raw_s <= 0:
                continue

            # Apply metadata filters
            if filters:
                meta = chunk.metadata
                match = True
                for f_key in ["category", "brand", "supplier", "country", "product_id"]:
                    f_val = filters.get(f_key)
                    if f_val is not None and str(f_val).strip():
                        chunk_val = getattr(meta, f_key, None)
                        if not chunk_val or str(chunk_val).strip().lower() != str(f_val).strip().lower():
                            match = False
                            break
                if not match:
                    continue

            # Normalize score to [0, 1]
            norm_score = raw_s / max_score
            scored_chunks.append((chunk, round(norm_score, 4)))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
