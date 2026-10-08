import logging
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.models import Chunk
from app.api.schemas import SourceItem

logger = logging.getLogger(__name__)


class ContextBuildResult(BaseModel):
    formatted_context: str
    sources: List[SourceItem]
    chunks_used: int
    total_chars: int


class ContextBuilder:
    """
    Builds LLM context from retrieved chunks.
    Enforces deduplication, relevance ordering, context character limits,
    and formats citations per Spec Section 7.1.
    """

    def __init__(self, max_context_chars: Optional[int] = None):
        settings = get_settings()
        self.max_context_chars = max_context_chars or settings.max_context_chars

    def deduplicate_chunks(
        self, chunks_with_scores: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        """
        Deduplicate chunks by chunk_id and near-identical content.
        Preserves the instance with the highest score.
        """
        seen_ids = set()
        seen_texts = set()
        deduped: List[Tuple[Chunk, float]] = []

        for chunk, score in chunks_with_scores:
            if chunk.chunk_id in seen_ids:
                continue

            # Normalized text signature (first 200 chars stripped of whitespace)
            clean_sig = "".join(chunk.text.split())[:200].lower()
            if clean_sig in seen_texts:
                continue

            seen_ids.add(chunk.chunk_id)
            seen_texts.add(clean_sig)
            deduped.append((chunk, score))

        return deduped

    def build_context(
        self, chunks_with_scores: List[Tuple[Chunk, float]]
    ) -> ContextBuildResult:
        """
        Format retrieved chunks into a prompt-ready context string and generate source list.
        """
        if not chunks_with_scores:
            return ContextBuildResult(
                formatted_context="",
                sources=[],
                chunks_used=0,
                total_chars=0,
            )

        # 1. Deduplicate
        deduped = self.deduplicate_chunks(chunks_with_scores)

        # 2. Sort by score descending (highest relevance first)
        deduped.sort(key=lambda x: x[1], reverse=True)

        context_blocks: List[str] = []
        sources: List[SourceItem] = []
        current_chars = 0
        chunks_used = 0

        for chunk, score in deduped:
            meta = chunk.metadata
            p_info = (
                f"{meta.product_name} ({meta.product_id})"
                if meta.product_name and meta.product_id
                else (meta.product_name or meta.product_id or "General")
            )

            # Block header with chunk ID for inline LLM citation (Spec 8.1)
            header = f"[Chunk {chunk.chunk_id} | Product: {p_info} | Doc: {meta.document} | Page: {meta.page} | Score: {score:.2f}]"
            block = f"{header}\n{chunk.text.strip()}\n"

            block_len = len(block)
            if current_chars + block_len > self.max_context_chars and chunks_used > 0:
                logger.info(
                    f"Context char limit reached ({current_chars}/{self.max_context_chars}). "
                    f"Truncating at {chunks_used} chunks."
                )
                break

            context_blocks.append(block)
            current_chars += block_len
            chunks_used += 1

            sources.append(
                SourceItem(
                    product_id=meta.product_id,
                    product_name=meta.product_name,
                    document=meta.document,
                    page=meta.page,
                    chunk_id=chunk.chunk_id,
                    source_type=meta.source_type,
                    score=round(score, 4),
                )
            )

        formatted_context = "\n---\n".join(context_blocks).strip()

        return ContextBuildResult(
            formatted_context=formatted_context,
            sources=sources,
            chunks_used=chunks_used,
            total_chars=len(formatted_context),
        )
