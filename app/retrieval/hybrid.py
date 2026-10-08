

import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

from app.config import get_settings
from app.ingestion.models import Chunk, ChunkMetadata
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.vector_store import VectorStore, ChromaVectorStore
from app.retrieval.bm25 import BM25Searcher
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import RetrievalResult

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStoreRetriever(BaseRetriever):
    """LangChain BaseRetriever adapter for VectorStore."""
    vector_store: Any
    embedder: Any
    k: int = 5
    filters: Optional[Dict[str, Any]] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_emb = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_embedding=query_emb,
            top_k=self.k,
            filters=self.filters,
        )
        return [chunk.to_document() for chunk, _ in results]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Chunk, float]]],
    k: int = 60,
    top_k: int = 5,
) -> List[Tuple[Chunk, float]]:
    """
    Combine multiple ranked chunk lists using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum(1 / (k + rank_i(d)))
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked_list, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)

    if not rrf_scores:
        return []

    # Maximum possible RRF score across len(ranked_lists) rankings is len(ranked_lists) / (k + 1)
    max_possible = len(ranked_lists) / (k + 1)

    fused: List[Tuple[Chunk, float]] = []
    for chunk_id, score in rrf_scores.items():
        chunk = chunk_map[chunk_id]
        normalized_score = min(1.0, score / max_possible)
        fused.append((chunk, round(normalized_score, 4)))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]


class HybridRetriever:
    """
    Hybrid Retriever combining Vector Semantic Search and BM25 Keyword Search
    using Reciprocal Rank Fusion (RRF, k=60), with multi-product comparison query splitting.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        bm25_searcher: Optional[BM25Searcher] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        max_context_chars: Optional[int] = None,
        rrf_k: int = 60,
    ):
        settings = get_settings()
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or get_embedder()
        self.top_k = top_k if top_k is not None else settings.top_k
        self.score_threshold = score_threshold if score_threshold is not None else settings.score_threshold
        self.rrf_k = rrf_k
        self.context_builder = ContextBuilder(max_context_chars=max_context_chars)

        # Initialize or load BM25 searcher
        if bm25_searcher:
            self.bm25_searcher = bm25_searcher
        else:
            self.bm25_searcher = BM25Searcher()
            self._init_bm25_from_vector_store()

    def _init_bm25_from_vector_store(self) -> None:
        """Load indexed chunks from vector store into BM25 index."""
        try:
            if isinstance(self.vector_store, ChromaVectorStore):
                collection = self.vector_store._collection
                count = collection.count()
                if count > 0:
                    results = collection.get()
                    ids = results.get("ids", [])
                    docs = results.get("documents", [])
                    metas = results.get("metadatas", [])

                    chunks = []
                    for i in range(len(ids)):
                        m = metas[i] if i < len(metas) else {}
                        chunk = Chunk(
                            chunk_id=ids[i],
                            text=docs[i],
                            metadata=ChunkMetadata(
                                product_id=m.get("product_id") or None,
                                product_name=m.get("product_name") or None,
                                category=m.get("category") or None,
                                brand=m.get("brand") or None,
                                supplier=m.get("supplier") or None,
                                country=m.get("country") or None,
                                document=m.get("document", ""),
                                page=int(m.get("page", 1)),
                                section=m.get("section") or None,
                                source_type=m.get("source_type", "text"),
                            ),
                        )
                        chunks.append(chunk)

                    self.bm25_searcher.index_chunks(chunks)
                    logger.info(f"Loaded {len(chunks)} chunks into BM25 index from vector store.")
        except Exception as e:
            logger.warning(f"Could not initialize BM25 from vector store: {e}")

    def get_ensemble_retriever(
        self,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[EnsembleRetriever]:
        """Create a LangChain EnsembleRetriever combining VectorStoreRetriever and BM25Retriever."""
        k = top_k or self.top_k
        bm25_lc = self.bm25_searcher.get_retriever(k=k * 2)
        if not bm25_lc:
            return None
        vec_lc = VectorStoreRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            k=k * 2,
            filters=filters,
        )
        return EnsembleRetriever(
            retrievers=[vec_lc, bm25_lc],
            weights=[0.5, 0.5],
            c=self.rrf_k,
            id_key="chunk_id",
        )

    def _detect_comparison_products(self, query: str) -> List[str]:
        """Detect multiple product mentions in a query for multi-query comparison retrieval."""
        known_products = [
            "SolarMax 550",
            "SolarMax 600",
            "SunPower Eco 400",
            "HelioCell 750",
            "TerraGrip Pro WorkBoot",
            "TerraGrip Ultra Lite",
            "TitanSteel HeavyDuty 900",
            "SafeStep Eco Runner",
            "SensorTech PT100",
            "OptiFlow Ultrasonic",
            "VibraSense Wireless",
            "P001",
            "P002",
            "P003",
            "P004",
            "P005",
            "P006",
            "P007",
            "P008",
            "P009",
            "P010",
            "P011",
            "P012",
        ]
        q_lower = query.lower()
        found = []
        for p in known_products:
            if p.lower() in q_lower:
                found.append(p)
        return found

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Execute Hybrid retrieval:
        1. If comparison query with 2+ products detected: split retrieval per product and merge.
        2. Vector search (top-2K)
        3. BM25 keyword search (top-2K)
        4. Reciprocal Rank Fusion (RRF, k=60)
        5. Filter by threshold & context building
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = score_threshold if score_threshold is not None else self.score_threshold

        clean_query = query.strip()
        if not clean_query:
            return RetrievalResult(query=query, has_context=False)

        # Multi-product comparison query handling (Spec Section 7.2)
        products_found = self._detect_comparison_products(clean_query)
        if len(products_found) >= 2 and any(w in clean_query.lower() for w in ["compare", "difference", "vs"]):
            logger.info(f"Comparison query detected with products: {products_found}. Retrieving per product.")
            per_product_chunks: List[Tuple[Chunk, float]] = []
            per_prod_k = max(2, effective_top_k // len(products_found) + 1)

            for prod in products_found:
                sub_query = f"{prod} technical specifications ratings warranty"
                sub_res = self._single_hybrid_retrieve(sub_query, filters=filters, top_k=per_prod_k)
                per_product_chunks.extend(sub_res)

            # Deduplicate by chunk_id
            seen = set()
            deduped_comparison = []
            for c, s in per_product_chunks:
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    deduped_comparison.append((c, s))

            fused_chunks = deduped_comparison[:effective_top_k]
        else:
            fused_chunks = self._single_hybrid_retrieve(clean_query, filters=filters, top_k=effective_top_k)

        # Apply score threshold (for hybrid RRF, threshold applies to normalized score)
        # Note: In hybrid RRF, max score is ~1.0; low matches are < 0.2
        # If user passed high threshold like 0.35, map appropriately:
        hybrid_threshold = min(0.35, effective_threshold * 0.5) if effective_threshold > 0.5 else effective_threshold
        passed_chunks = [item for item in fused_chunks if item[1] >= hybrid_threshold]

        if not passed_chunks:
            logger.info(f"No chunks passed hybrid threshold for query: '{clean_query[:50]}...'")
            return RetrievalResult(
                query=query,
                chunks_with_scores=[],
                context="",
                sources=[],
                has_context=False,
                filters_applied=filters,
            )

        build_res = self.context_builder.build_context(passed_chunks)

        return RetrievalResult(
            query=query,
            chunks_with_scores=passed_chunks,
            context=build_res.formatted_context,
            sources=build_res.sources,
            has_context=bool(build_res.formatted_context),
            filters_applied=filters,
        )

    def _single_hybrid_retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """Run vector + BM25 and combine with RRF."""
        # 1. Vector Search
        query_emb = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(
            query_embedding=query_emb,
            top_k=top_k * 2,
            filters=filters,
        )

        # 2. BM25 Search
        bm25_results = self.bm25_searcher.search(
            query=query,
            top_k=top_k * 2,
            filters=filters,
        )

        # 3. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            ranked_lists=[vector_results, bm25_results],
            k=self.rrf_k,
            top_k=top_k,
        )

        return fused
