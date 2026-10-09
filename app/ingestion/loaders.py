import io
import re
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import pymupdf

from app.config import get_settings
from app.ingestion.models import RawDocumentPage
from app.ingestion.ocr import ocr_image, ocr_image_bytes, ocr_pixmap, is_ocr_available, is_tesseract_available
from app.ingestion.normalizer import clean_text

logger = logging.getLogger(__name__)

Tuple_ProdInfo = tuple[Optional[str], Optional[str]]


def infer_product_info_from_filename(filename: str, catalog: Dict[str, Dict[str, Any]]) -> Tuple_ProdInfo:
    """Infer product_id and product_name from filename or catalog match."""
    stem = Path(filename).stem.lower()

    # Direct match on product_id in filename (e.g. P001, P012)
    p_match = re.search(r'\b(p\d{3})\b', stem)
    if p_match:
        pid = p_match.group(1).upper()
        if pid in catalog:
            return pid, catalog[pid].get("product_name")

    # Match product name words in catalog
    for pid, entry in catalog.items():
        pname = entry.get("product_name", "")
        clean_name = re.sub(r'[^a-z0-9]', '', pname.lower())
        clean_stem = re.sub(r'[^a-z0-9]', '', stem)
        if clean_name in clean_stem or clean_stem in clean_name:
            return pid, pname

    return None, None


class DocumentLoaderRegistry:
    """Dispatches loaders by file extension and formats raw pages."""

    def __init__(self, catalog: Optional[Dict[str, Dict[str, Any]]] = None):
        self.catalog = catalog or {}
        self.settings = get_settings()

    def load_catalog(self, catalog_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load catalog from products.json or products.csv."""
        ext = catalog_path.suffix.lower()
        catalog: Dict[str, Dict[str, Any]] = {}

        if ext == ".json":
            with open(catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    pid = item.get("product_id")
                    if pid:
                        catalog[pid] = item
        elif ext == ".csv":
            with open(catalog_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for item in reader:
                    pid = item.get("product_id")
                    if pid:
                        catalog[pid] = item

        self.catalog.update(catalog)
        return self.catalog

    def load_file(self, file_path: Path) -> List[RawDocumentPage]:
        """Dispatch file loading based on extension."""
        ext = file_path.suffix.lower()
        if ext in [".json", ".csv"]:
            return self._load_structured(file_path)
        elif ext in [".md", ".markdown", ".txt"]:
            return self._load_text_file(file_path)
        elif ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
            return self._load_image(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path.name}")
            return []

    def _load_structured(self, file_path: Path) -> List[RawDocumentPage]:
        """Convert structured records into readable spec sheet pages."""
        pages: List[RawDocumentPage] = []
        ext = file_path.suffix.lower()
        records: List[Dict[str, Any]] = []

        if ext == ".json":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        records = content
                    elif isinstance(content, dict):
                        records = [content]
            except Exception as e:
                logger.error(f"Error reading JSON {file_path}: {e}")
                raise
        elif ext == ".csv":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    records = list(reader)
            except Exception as e:
                logger.error(f"Error reading CSV {file_path}: {e}")
                raise

        for idx, rec in enumerate(records, start=1):
            pid = rec.get("product_id")
            pname = rec.get("product_name")

            # Update catalog
            if pid:
                self.catalog[pid] = rec

            lines = ["Product Specification Record:"]
            for k, v in rec.items():
                if v is not None and str(v).strip():
                    key_fmt = k.replace("_", " ").title()
                    lines.append(f"{key_fmt}: {v}")

            spec_text = "\n".join(lines)
            cleaned = clean_text(spec_text)

            pages.append(
                RawDocumentPage(
                    document_name=file_path.name,
                    page_number=idx,
                    text=cleaned,
                    source_type="structured",
                    inferred_product_id=pid,
                    inferred_product_name=pname,
                    extra_metadata=rec,
                )
            )

        return pages

    def _load_text_file(self, file_path: Path) -> List[RawDocumentPage]:
        """Load Markdown or plain text documents."""
        try:
            raw = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = file_path.read_text(encoding="latin-1", errors="replace")

        cleaned = clean_text(raw)
        pid, pname = infer_product_info_from_filename(file_path.name, self.catalog)

        # Also inspect top of text for explicit product references
        if not pid:
            m = re.search(r'\b(P\d{3})\b', cleaned[:300])
            if m and m.group(1) in self.catalog:
                pid = m.group(1)
                pname = self.catalog[pid].get("product_name")

        return [
            RawDocumentPage(
                document_name=file_path.name,
                page_number=1,
                text=cleaned,
                source_type="text",
                inferred_product_id=pid,
                inferred_product_name=pname,
            )
        ]

    def _load_pdf(self, file_path: Path) -> List[RawDocumentPage]:
        """Load PDF with automatic OCR fallback for scanned pages and embedded image OCR for hybrid pages."""
        pages: List[RawDocumentPage] = []
        doc = pymupdf.open(str(file_path))
        pid, pname = infer_product_info_from_filename(file_path.name, self.catalog)

        for p_idx, page in enumerate(doc, start=1):
            extracted = page.get_text().strip()
            cleaned = clean_text(extracted)

            is_scanned = (
                self.settings.ocr_force_pdf_ocr
                or len(cleaned) < self.settings.ocr_min_text_chars
            )

            if not is_scanned:
                # Text-based page: check if there are embedded scanned images
                page_pid = pid
                page_pname = pname
                if not page_pid:
                    m = re.search(r'\b(P\d{3})\b', cleaned[:300])
                    if m and m.group(1) in self.catalog:
                        page_pid = m.group(1)
                        page_pname = self.catalog[page_pid].get("product_name")

                combined_text = cleaned
                embedded_confs: List[float] = []

                if self.settings.ocr_extract_embedded_images:
                    images = page.get_images(full=True)
                    for img_idx, img_meta in enumerate(images, start=1):
                        xref = img_meta[0]
                        try:
                            base_img = doc.extract_image(xref)
                        except Exception as e:
                            logger.debug(f"Failed to extract image xref {xref} from {file_path.name}: {e}")
                            continue

                        w = base_img.get("width", 0)
                        h = base_img.get("height", 0)
                        if w < self.settings.ocr_min_image_dim or h < self.settings.ocr_min_image_dim:
                            continue

                        img_text, conf, low = ocr_image_bytes(base_img["image"])
                        cleaned_img = clean_text(img_text)
                        if cleaned_img:
                            logger.info(
                                f"Extracted OCR text from embedded image {img_idx} on page {p_idx} of '{file_path.name}' "
                                f"({len(cleaned_img)} chars, conf={conf}%)"
                            )
                            combined_text += f"\n\n### [Embedded Scanned Image: Figure {img_idx} on Page {p_idx}]\n{cleaned_img}"
                            if conf is not None:
                                embedded_confs.append(conf)

                mean_conf = (sum(embedded_confs) / len(embedded_confs)) if embedded_confs else None

                pages.append(
                    RawDocumentPage(
                        document_name=file_path.name,
                        page_number=p_idx,
                        text=combined_text,
                        source_type="text",
                        ocr_confidence=round(mean_conf, 2) if mean_conf is not None else None,
                        low_confidence=False,
                        inferred_product_id=page_pid,
                        inferred_product_name=page_pname,
                    )
                )
            else:
                # Scanned or image-only page: rasterize at configured DPI and OCR
                logger.info(
                    f"Page {p_idx} of '{file_path.name}' treated as scanned ({len(cleaned)} chars). Running OCR..."
                )
                ocr_text = ""
                mean_conf = None
                is_low_conf = True

                # Attempt 1: Direct extraction if page has a single full-page image
                images = page.get_images(full=True)
                if len(images) == 1:
                    try:
                        base_img = doc.extract_image(images[0][0])
                        ocr_text, mean_conf, is_low_conf = ocr_image_bytes(base_img["image"])
                    except Exception as e:
                        logger.debug(f"Direct image extraction failed for page {p_idx}: {e}")

                # Attempt 2: Rasterize page at configured DPI if direct extraction yielded no text
                if not ocr_text:
                    pix = page.get_pixmap(dpi=self.settings.ocr_dpi)
                    ocr_text, mean_conf, is_low_conf = ocr_pixmap(pix)

                cleaned_ocr = clean_text(ocr_text)

                page_pid = pid
                page_pname = pname
                if not page_pid and cleaned_ocr:
                    m = re.search(r'\b(P\d{3})\b', cleaned_ocr[:300])
                    if m and m.group(1) in self.catalog:
                        page_pid = m.group(1)
                        page_pname = self.catalog[page_pid].get("product_name")

                pages.append(
                    RawDocumentPage(
                        document_name=file_path.name,
                        page_number=p_idx,
                        text=cleaned_ocr,
                        source_type="ocr",
                        ocr_confidence=mean_conf,
                        low_confidence=is_low_conf,
                        inferred_product_id=page_pid,
                        inferred_product_name=page_pname,
                    )
                )

        doc.close()
        return pages

    def _load_image(self, file_path: Path) -> List[RawDocumentPage]:
        """Load standalone image file and run OCR."""
        logger.info(f"Loading standalone image '{file_path.name}' for OCR...")
        img = Image.open(file_path)
        ocr_text, mean_conf, is_low_conf = ocr_image(img)
        cleaned_ocr = clean_text(ocr_text)

        pid, pname = infer_product_info_from_filename(file_path.name, self.catalog)
        if not pid and cleaned_ocr:
            m = re.search(r'\b(P\d{3})\b', cleaned_ocr[:300])
            if m and m.group(1) in self.catalog:
                pid = m.group(1)
                pname = self.catalog[pid].get("product_name")

        return [
            RawDocumentPage(
                document_name=file_path.name,
                page_number=1,
                text=cleaned_ocr,
                source_type="ocr",
                ocr_confidence=mean_conf,
                low_confidence=is_low_conf,
                inferred_product_id=pid,
                inferred_product_name=pname,
            )
        ]

