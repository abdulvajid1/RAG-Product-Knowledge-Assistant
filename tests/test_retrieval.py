import pytest
from pathlib import Path
from typing import List

from app.ingestion.models import Chunk, ChunkMetadata
from app.retrieval.embedder import MockEmbedder
from app.retrieval.vector_store import ChromaVectorStore
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import BaselineRetriever


@pytest.fixture
def populated_vector_store(tmp_path):
    """Create an ephemeral vector store with synthetic test chunks across categories and metadata."""
    store_dir = tmp_path / "test_chroma"
    store = ChromaVectorStore(persist_directory=str(store_dir))
    embedder = MockEmbedder(dimension=384)

    test_chunks = [
        Chunk(
            chunk_id="P001_01",
            text="[Product: SolarMax 550 (P001) | Document: solarmax.pdf | Page: 1]\nMaximum operating temperature is 85°C. Power output is 550W.",
            metadata=ChunkMetadata(
                product_id="P001",
                product_name="SolarMax 550",
                category="Solar Panels",
                brand="SolarMax",
                supplier="Vikram Solar Energies Pvt Ltd",
                country="India",
                document="solarmax.pdf",
                page=1,
                source_type="text",
            ),
        ),
        Chunk(
            chunk_id="P002_01",
            text="[Product: SolarMax 600 (P002) | Document: solarmax_600.pdf | Page: 1]\nMaximum operating temperature is 85°C. Power output is 600W bifacial TOPCon.",
            metadata=ChunkMetadata(
                product_id="P002",
                product_name="SolarMax 600",
                category="Solar Panels",
                brand="SolarMax",
                supplier="Vikram Solar Energies Pvt Ltd",
                country="India",
                document="solarmax_600.pdf",
                page=1,
                source_type="text",
            ),
        ),
        Chunk(
            chunk_id="P005_01",
            text="[Product: TerraGrip Pro WorkBoot (P005) | Document: terragrip.md | Page: 1]\nWaterproof heavy-duty construction safety boot with 200J steel toe cap and Kevlar midsole.",
            metadata=ChunkMetadata(
                product_id="P005",
                product_name="TerraGrip Pro WorkBoot",
                category="Safety Shoes",
                brand="TerraGrip",
                supplier="Acme Industrial Safety",
                country="India",
                document="terragrip.md",
                page=1,
                source_type="text",
            ),
        ),
        Chunk(
            chunk_id="P007_01",
            text="[Product: TitanSteel HeavyDuty 900 (P007) | Document: titansteel.txt | Page: 1]\nGerman safety boots with waterproof Sympatex lining, steel toe, 300°C heat-resistant outsole.",
            metadata=ChunkMetadata(
                product_id="P007",
                product_name="TitanSteel HeavyDuty 900",
                category="Safety Shoes",
                brand="TitanSteel",
                supplier="Rheinland Workwear GmbH",
                country="Germany",
                document="titansteel.txt",
                page=1,
                source_type="text",
            ),
        ),
    ]

    texts = [c.text for c in test_chunks]
    embeddings = embedder.embed_documents(texts)
    store.add_chunks(test_chunks, embeddings)

    return store, embedder, test_chunks


def test_context_builder_deduplication():
    """Verify ContextBuilder dedupes identical chunk IDs and normalized texts."""
    builder = ContextBuilder(max_context_chars=5000)

    chunk1 = Chunk(
        chunk_id="C001",
        text="SolarMax 550 operating temperature is 85°C.",
        metadata=ChunkMetadata(document="doc1.pdf", page=1),
    )
    # Duplicate ID
    chunk2 = Chunk(
        chunk_id="C001",
        text="SolarMax 550 operating temperature is 85°C.",
        metadata=ChunkMetadata(document="doc1.pdf", page=1),
    )
    # Duplicate text with different ID
    chunk3 = Chunk(
        chunk_id="C002",
        text="SolarMax 550 operating temperature is 85°C.",
        metadata=ChunkMetadata(document="doc2.pdf", page=2),
    )

    result = builder.build_context([
        (chunk1, 0.85),
        (chunk2, 0.80),
        (chunk3, 0.75),
    ])

    assert result.chunks_used == 1
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "C001"
    assert result.sources[0].score == 0.85


def test_context_builder_budget_cap():
    """Verify ContextBuilder respects max_context_chars budget cap."""
    builder = ContextBuilder(max_context_chars=250)

    chunk1 = Chunk(
        chunk_id="C001",
        text="A" * 150,
        metadata=ChunkMetadata(document="doc1.pdf", page=1),
    )
    chunk2 = Chunk(
        chunk_id="C002",
        text="B" * 150,
        metadata=ChunkMetadata(document="doc2.pdf", page=1),
    )

    result = builder.build_context([(chunk1, 0.90), (chunk2, 0.85)])
    assert result.chunks_used == 1
    assert len(result.sources) == 1
    assert "Chunk C001" in result.formatted_context
    assert "Chunk C002" not in result.formatted_context


def test_retriever_top_k(populated_vector_store):
    """Verify top_k parameter limits retrieved results."""
    store, embedder, _ = populated_vector_store
    retriever = BaselineRetriever(
        vector_store=store,
        embedder=embedder,
        top_k=2,
        score_threshold=0.0,
    )

    res = retriever.retrieve("SolarMax panel operating temperature", top_k=2)
    assert res.has_context is True
    assert len(res.chunks_with_scores) <= 2
    assert len(res.sources) <= 2


def test_retriever_score_threshold_no_result(populated_vector_store):
    """Verify retriever returns empty context when chunks do not pass the score threshold."""
    store, embedder, _ = populated_vector_store
    # Set impossible threshold (0.99)
    retriever = BaselineRetriever(
        vector_store=store,
        embedder=embedder,
        top_k=5,
        score_threshold=0.99,
    )

    res = retriever.retrieve("random unrelated query")
    assert res.has_context is False
    assert res.context == ""
    assert res.sources == []
    assert len(res.chunks_with_scores) == 0


def test_metadata_filters_individually(populated_vector_store):
    """Verify all 5 required metadata filters: category, brand, supplier, country, product_id."""
    store, embedder, _ = populated_vector_store
    retriever = BaselineRetriever(
        vector_store=store,
        embedder=embedder,
        top_k=10,
        score_threshold=0.0,
    )

    # 1. Category filter
    res_cat = retriever.retrieve("waterproof boots", filters={"category": "Safety Shoes"})
    assert res_cat.has_context is True
    assert all(s.product_id in ["P005", "P007"] for s in res_cat.sources)

    # 2. Country filter
    res_country = retriever.retrieve("safety equipment", filters={"country": "Germany"})
    assert res_country.has_context is True
    assert all(s.product_id == "P007" for s in res_country.sources)

    # 3. Brand filter
    res_brand = retriever.retrieve("solar modules", filters={"brand": "SolarMax"})
    assert res_brand.has_context is True
    assert all(s.product_id in ["P001", "P002"] for s in res_brand.sources)

    # 4. Supplier filter
    res_supplier = retriever.retrieve("products", filters={"supplier": "Vikram Solar Energies Pvt Ltd"})
    assert res_supplier.has_context is True
    assert all(s.product_id in ["P001", "P002"] for s in res_supplier.sources)

    # 5. Product_id filter
    res_pid = retriever.retrieve("temperature", filters={"product_id": "P001"})
    assert res_pid.has_context is True
    assert len(res_pid.sources) == 1
    assert res_pid.sources[0].product_id == "P001"


def test_combined_metadata_filters(populated_vector_store):
    """Verify combined metadata filters work correctly."""
    store, embedder, _ = populated_vector_store
    retriever = BaselineRetriever(
        vector_store=store,
        embedder=embedder,
        top_k=10,
        score_threshold=0.0,
    )

    # Category: Safety Shoes AND Country: India -> only P005 (TerraGrip)
    res_comb = retriever.retrieve(
        "boots", filters={"category": "Safety Shoes", "country": "India"}
    )
    assert res_comb.has_context is True
    assert len(res_comb.sources) == 1
    assert res_comb.sources[0].product_id == "P005"
