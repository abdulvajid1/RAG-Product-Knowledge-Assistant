import io
import fitz
import pytest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from app.config import get_settings
from app.ingestion.models import RawDocumentPage
from app.ingestion.loaders import DocumentLoaderRegistry
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.ocr import is_ocr_available, ocr_image


def get_test_font(size: int = 24):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def create_test_image(text: str, width: int = 600, height: int = 200) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = get_test_font(24)
    draw.text((20, 40), text, fill=(0, 0, 0), font=font)
    return img


@pytest.fixture
def test_catalog():
    return {
        "P001": {"product_id": "P001", "product_name": "SolarMax 550", "category": "Solar Panels"},
        "P005": {"product_id": "P005", "product_name": "TerraGrip Pro WorkBoot", "category": "Safety Shoes"},
        "P010": {"product_id": "P010", "product_name": "SensorTech TempPro PT100", "category": "Industrial Sensors"},
    }


def test_pure_scanned_pdf_ingestion(tmp_path, test_catalog):
    """Verify that an image-only PDF with no text layer is recognized as scanned and OCR'd."""
    if not is_ocr_available():
        pytest.skip("No OCR engine available on system.")

    pdf_path = tmp_path / "scanned_test.pdf"
    img = create_test_image("SolarMax 550 High Efficiency Module", 800, 300)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=800, height=300)
    page.insert_image(page.rect, stream=img_bytes.getvalue())
    doc.save(str(pdf_path))
    doc.close()

    registry = DocumentLoaderRegistry(catalog=test_catalog)
    pages = registry.load_file(pdf_path)

    assert len(pages) == 1
    page = pages[0]
    assert page.source_type == "ocr"
    assert "SolarMax" in page.text or "550" in page.text
    assert page.ocr_confidence is not None
    assert page.ocr_confidence > 0


def test_hybrid_pdf_ingestion_with_embedded_image(tmp_path, test_catalog):
    """Verify that a PDF containing digital text AND an embedded scanned image extracts both."""
    if not is_ocr_available():
        pytest.skip("No OCR engine available on system.")

    pdf_path = tmp_path / "hybrid_spec_sheet.pdf"

    # Create embedded image
    cert_img = create_test_image("CERTIFICATION IEC 61215 PASSED", 600, 200)
    cert_bytes = io.BytesIO()
    cert_img.save(cert_bytes, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Insert digital text
    page.insert_text(
        fitz.Point(50, 50),
        "SolarMax 550 Technical Sheet\nThis is native selectable digital text describing specifications.",
        fontsize=12,
    )
    # Insert embedded image figure
    img_rect = fitz.Rect(50, 150, 450, 300)
    page.insert_image(img_rect, stream=cert_bytes.getvalue())
    doc.save(str(pdf_path))
    doc.close()

    registry = DocumentLoaderRegistry(catalog=test_catalog)
    pages = registry.load_file(pdf_path)

    assert len(pages) == 1
    page = pages[0]
    # Page contains digital text
    assert "SolarMax 550" in page.text
    # Page also contains extracted text from embedded image
    assert "CERTIFICATION" in page.text or "61215" in page.text or "PASSED" in page.text
    assert "Embedded Scanned Image" in page.text

    # Verify chunker creates dedicated OCR chunk for the embedded figure
    chunker = StructureAwareChunker(catalog=test_catalog)
    chunks = chunker.chunk_page(page)
    assert len(chunks) >= 2

    ocr_chunks = [c for c in chunks if c.metadata.source_type == "ocr"]
    assert len(ocr_chunks) >= 1
    assert "CERTIFICATION" in ocr_chunks[0].text or "61215" in ocr_chunks[0].text or "PASSED" in ocr_chunks[0].text


def test_embedded_image_dimension_filtering(tmp_path, test_catalog):
    """Verify that tiny icons or bullets (< 100x100) are skipped and not OCR'd."""
    pdf_path = tmp_path / "tiny_icon_test.pdf"

    # Tiny 30x30 icon image
    icon_img = Image.new("RGB", (30, 30), color=(100, 100, 100))
    icon_bytes = io.BytesIO()
    icon_img.save(icon_bytes, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 50), "Document with plenty of text to exceed min chars threshold.", fontsize=12)
    page.insert_image(fitz.Rect(50, 100, 80, 130), stream=icon_bytes.getvalue())
    doc.save(str(pdf_path))
    doc.close()

    registry = DocumentLoaderRegistry(catalog=test_catalog)
    pages = registry.load_file(pdf_path)

    assert len(pages) == 1
    # Tiny icon should not add an embedded scanned image section
    assert "Embedded Scanned Image" not in pages[0].text


def test_real_raw_scanned_documents():
    """Verify that sample scanned PDFs in data/raw are successfully loaded and OCR'd."""
    if not is_ocr_available():
        pytest.skip("No OCR engine available on system.")

    registry = DocumentLoaderRegistry()
    registry.load_catalog(Path("data/raw/products.json"))

    cert_path = Path("data/raw/scanned_terragrip_pro_cert.pdf")
    if cert_path.exists():
        pages = registry.load_file(cert_path)
        assert len(pages) == 1
        assert pages[0].source_type == "ocr"
        assert len(pages[0].text) > 50
        assert "LAB-IN-2026-9941" in pages[0].text or "TerraGrip" in pages[0].text
        assert pages[0].inferred_product_id == "P005"

    calib_path = Path("data/raw/scanned_sensortech_pt100_calib.pdf")
    if calib_path.exists():
        pages = registry.load_file(calib_path)
        assert len(pages) == 1
        assert pages[0].source_type == "ocr"
        assert len(pages[0].text) > 50
        assert "CAL-IND-88204" in pages[0].text or "SensorTech" in pages[0].text
        assert pages[0].inferred_product_id == "P010"
