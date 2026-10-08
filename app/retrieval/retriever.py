import logging
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.models import Chunk
from app.api.schemas import SourceItem
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.vector_store import VectorStore, ChromaVectorStore
from app.retrieval.context_builder import ContextBuilder, ContextBuildResult

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    query: str
    chunks_with_scores: List[Tuple[Chunk, float]] = Field(default_factory=list)
    context: str = ""
    sources: List[SourceItem] = Field(default_factory=list)
    has_context: bool = False
    filters_applied: Optional[Dict[str, Any]] = None


class BaselineRetriever:
    """
    Baseline Vector-based Retriever.
    Orchestrates query embedding, vector search, metadata pre-filtering,
    similarity score thresholding, and grounded context construction.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        max_context_chars: Optional[int] = None,
    ):
        settings = get_settings()
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or get_embedder()
        self.top_k = top_k if top_k is not None else settings.top_k
        self.score_threshold = score_threshold if score_threshold is not None else settings.score_threshold
        self.context_builder = ContextBuilder(max_context_chars=max_context_chars)

    def as_langchain_retriever(
        self,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """Return a LangChain BaseRetriever adapter."""
        from app.retrieval.hybrid import VectorStoreRetriever
        return VectorStoreRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            k=top_k or self.top_k,
            filters=filters,
        )

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Execute baseline vector retrieval pipeline:
        query -> embed -> vector search (top-K) -> metadata filter -> score threshold -> context builder
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = score_threshold if score_threshold is not None else self.score_threshold

        clean_query = query.strip()
        if not clean_query:
            return RetrievalResult(query=query, has_context=False)

        # 1. Embed query
        query_embedding = self.embedder.embed_query(clean_query)

        # 2. Vector search with pre-filtering
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            filters=filters,
        )

        logger.info(
            f"Retrieved {len(raw_results)} chunks from vector store for query: '{clean_query[:50]}...'"
        )

        # 3. Apply score threshold
        passed_chunks: List[Tuple[Chunk, float]] = []
        for chunk, score in raw_results:
            if score >= effective_threshold:
                passed_chunks.append((chunk, score))
            else:
                logger.debug(
                    f"Chunk '{chunk.chunk_id}' score {score:.4f} below threshold {effective_threshold:.4f} - dropped"
                )

        # 4. Handle No-result case (Spec 7.1: return empty context if nothing passes threshold)
        if not passed_chunks:
            logger.info(
                f"No chunks passed score threshold ({effective_threshold}) for query: '{clean_query[:50]}...'"
            )
            return RetrievalResult(
                query=query,
                chunks_with_scores=[],
                context="",
                sources=[],
                has_context=False,
                filters_applied=filters,
            )

        # 5. Build context & sources
        build_result = self.context_builder.build_context(passed_chunks)

        return RetrievalResult(
            query=query,
            chunks_with_scores=passed_chunks,
            context=build_result.formatted_context,
            sources=build_result.sources,
            has_context=bool(build_result.formatted_context),
            filters_applied=filters,
        )


_cached_retriever: Optional[Any] = None


def get_retriever(force_mode: Optional[str] = None) -> Any:
    """Singleton getter for application retriever, switchable via RETRIEVAL_MODE config."""
    global _cached_retriever
    settings = get_settings()
    mode = (force_mode or settings.retrieval_mode).lower()

    if _cached_retriever is not None and not force_mode:
        return _cached_retriever

    if mode == "hybrid":
        from app.retrieval.hybrid import HybridRetriever
        retriever = HybridRetriever()
    else:
        retriever = BaselineRetriever()

    if not force_mode:
        _cached_retriever = retriever

    return retriever
