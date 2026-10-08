import os
import json
import csv
import pytest
from pathlib import Path
from PIL import Image, ImageDraw

from app.ingestion.models import RawDocumentPage, Chunk, ChunkMetadata
from app.ingestion.loaders import DocumentLoaderRegistry
from app.ingestion.chunker import StructureAwareChunker, generate_deterministic_chunk_id, split_text_by_headings
from app.ingestion.normalizer import clean_text, normalize_units, fix_hyphenated_linebreaks
from app.ingestion.ocr import is_tesseract_available, ocr_image
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embedder import MockEmbedder
from app.retrieval.vector_store import ChromaVectorStore


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with test fixtures."""
    d = tmp_path / "data"
    d.mkdir()

    # 1. products.json
    products = [
        {
            "product_id": "P001",
            "product_name": "SolarMax 550",
            "category": "Solar Panels",
            "brand": "SolarMax",
            "supplier": "Vikram Solar Energies Pvt Ltd",
            "country": "India",
            "price": 220.0,
            "short_description": "550W high efficiency solar panel",
        },
        {
            "product_id": "P005",
            "product_name": "TerraGrip Pro WorkBoot",
            "category": "Safety Shoes",
            "brand": "TerraGrip",
            "supplier": "Acme Industrial Safety",
            "country": "India",
            "price": 65.0,
            "short_description": "Waterproof steel toe boot",
        },
    ]
    with open(d / "products.json", "w", encoding="utf-8") as f:
        json.dump(products, f)

    # 2. products.csv
    with open(d / "products.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(products[0].keys()))
        writer.writeheader()
        writer.writerows(products)

    # 3. sample.md
    md_content = """# SolarMax 550 Technical Sheet

## Overview
SolarMax 550 is a top-tier commercial solar module.

## Specifications
- Operating temperature: -40°C to 85°C
- Maximum Power: 550W
- Warranty: 25 years
"""
    (d / "solarmax_550.md").write_text(md_content, encoding="utf-8")

    # 4. sample.txt
    txt_content = """TerraGrip Pro WorkBoot
Specifications:
Steel toe cap: 200 Joules
Water resistance: S3 certified
Midsole: Puncture resistant Kevlar
"""
    (d / "terragrip_boot.txt").write_text(txt_content, encoding="utf-8")

    return d


def test_normalizer_rules():
    """Verify clean_text normalizes units, unicode, whitespace, and hyphenated linebreaks."""
    raw = "The operat-\ning temperature is 85 °C and power is 550 W.   Page 1 of 2\n\n\n\nNext line."
    cleaned = clean_text(raw)
    assert "operating temperature" in cleaned
    assert "85°C" in cleaned
    assert "550W" in cleaned
    assert "Page 1 of 2" not in cleaned
    assert "\n\n\n" not in cleaned


def test_structured_loader_json_and_csv(temp_data_dir):
    """Verify structured loaders create records with source_type='structured' and metadata."""
    registry = DocumentLoaderRegistry()
    registry.load_catalog(temp_data_dir / "products.json")

    json_pages = registry.load_file(temp_data_dir / "products.json")
    assert len(json_pages) == 2
    assert json_pages[0].source_type == "structured"
    assert json_pages[0].inferred_product_id == "P001"
    assert json_pages[0].inferred_product_name == "SolarMax 550"
    assert "SolarMax 550" in json_pages[0].text

    csv_pages = registry.load_file(temp_data_dir / "products.csv")
    assert len(csv_pages) == 2
    assert csv_pages[1].source_type == "structured"
    assert csv_pages[1].inferred_product_id == "P005"


def test_text_and_markdown_loaders(temp_data_dir):
    """Verify Markdown and plain text loaders attach inferred product info and source_type='text'."""
    registry = DocumentLoaderRegistry()
    registry.load_catalog(temp_data_dir / "products.json")

    md_pages = registry.load_file(temp_data_dir / "solarmax_550.md")
    assert len(md_pages) == 1
    assert md_pages[0].source_type == "text"
    assert md_pages[0].inferred_product_id == "P001"
    assert "SolarMax 550" in md_pages[0].text

    txt_pages = registry.load_file(temp_data_dir / "terragrip_boot.txt")
    assert len(txt_pages) == 1
    assert txt_pages[0].source_type == "text"
    assert "TerraGrip" in txt_pages[0].text


def test_structure_aware_chunking(temp_data_dir):
    """Verify structure-aware chunker preserves section headers, prefixes, and generates valid schema."""
    catalog = {
        "P001": {
            "product_id": "P001",
            "product_name": "SolarMax 550",
            "category": "Solar Panels",
            "brand": "SolarMax",
            "supplier": "Vikram Solar",
            "country": "India",
        }
    }
    chunker = StructureAwareChunker(catalog=catalog)

    page = RawDocumentPage(
        document_name="solarmax_550.md",
        page_number=1,
        text="## Specifications\nMaximum power is 550W.\nOperating temperature is 85°C.",
        source_type="text",
        inferred_product_id="P001",
        inferred_product_name="SolarMax 550",
    )

    chunks = chunker.chunk_page(page)
    assert len(chunks) >= 1
    chunk = chunks[0]

    # Context header check per Spec 6.2
    assert "[Product: SolarMax 550 (P001) | Document: solarmax_550.md | Page: 1 | Section: Specifications]" in chunk.text
    assert chunk.metadata.product_id == "P001"
    assert chunk.metadata.category == "Solar Panels"
    assert chunk.metadata.source_type == "text"
    assert chunk.metadata.document == "solarmax_550.md"
    assert chunk.metadata.page == 1
    assert chunk.chunk_id.startswith("P001_")


def test_deterministic_chunk_id():
    """Verify chunk IDs are deterministic and non-random."""
    id1 = generate_deterministic_chunk_id("P001", "datasheet.pdf", 2, 3, "Some text")
    id2 = generate_deterministic_chunk_id("P001", "datasheet.pdf", 2, 3, "Some text")
    assert id1 == id2
    assert id1 == "P001_DATASHEET_P2_03"


def test_idempotent_ingestion(tmp_path, temp_data_dir):
    """Verify that running ingestion twice does not create duplicate chunks in vector store."""
    index_dir = tmp_path / "vector_store"
    vector_store = ChromaVectorStore(persist_directory=str(index_dir))
    embedder = MockEmbedder(dimension=384)

    pipeline = IngestionPipeline(vector_store=vector_store, embedder=embedder)

    # First run
    stats1 = pipeline.run(data_dir=temp_data_dir, reset=True)
    count1 = vector_store.count()
    assert count1 > 0
    assert stats1["chunks_indexed"] == count1

    # Second run without reset (upsert should not duplicate IDs)
    stats2 = pipeline.run(data_dir=temp_data_dir, reset=False)
    count2 = vector_store.count()
    assert count2 == count1, f"Expected {count1} chunks, but got {count2} after second ingestion"


def test_resilience_to_bad_file(tmp_path, temp_data_dir):
    """Verify a malformed file is logged as a failure without crashing the whole pipeline."""
    bad_file = temp_data_dir / "corrupted.json"
    bad_file.write_text("{This is not valid JSON!}", encoding="utf-8")

    index_dir = tmp_path / "vector_store_resilience"
    vector_store = ChromaVectorStore(persist_directory=str(index_dir))
    embedder = MockEmbedder(dimension=384)

    pipeline = IngestionPipeline(vector_store=vector_store, embedder=embedder)
    stats = pipeline.run(data_dir=temp_data_dir, reset=True)

    assert stats["files_processed"] >= 3
    assert stats["chunks_indexed"] > 0
    # The bad json file should be logged under failures
    failed_files = [f["file"] for f in stats["failures"]]
    assert "corrupted.json" in failed_files


def test_ocr_handling(tmp_path):
    """Verify OCR execution if Tesseract is installed, or skip per Spec 12 if absent."""
    if not is_tesseract_available():
        pytest.skip("Tesseract OCR binary not installed on system path; skipping per Spec 12.")

    # Create synthetic test image with clean text
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "SolarMax 550 Rating 85C", fill=(0, 0, 0))

    text, conf, low_conf = ocr_image(img)
    assert len(text) > 0
    assert "SolarMax" in text or "550" in text
    assert conf is not None
    assert conf > 0
