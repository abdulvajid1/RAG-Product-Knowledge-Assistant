import re
import hashlib
from typing import List, Dict, Optional, Any
from app.ingestion.models import Chunk, ChunkMetadata, RawDocumentPage
from app.config import get_settings


from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def generate_deterministic_chunk_id(
    product_id: Optional[str],
    doc_name: str,
    page_num: int,
    chunk_index: int,
    chunk_text: str = "",
) -> str:
    """Generate a deterministic, readable chunk ID."""
    prefix = product_id.strip() if product_id else "DOC"
    path = Path(doc_name)
    stem = re.sub(r'[^a-zA-Z0-9]', '', path.stem)[:10].upper()
    ext = re.sub(r'[^a-zA-Z0-9]', '', path.suffix)[:3].upper()
    doc_tag = f"{stem}_{ext}" if ext and ext != "PDF" else stem
    return f"{prefix}_{doc_tag}_P{page_num}_{chunk_index:02d}"


def split_text_by_headings(text: str) -> List[Dict[str, str]]:
    """
    Split text into logical sections based on markdown headings or prominent dividers.
    Returns list of dicts with 'section' title and 'content'.
    """
    lines = text.split("\n")
    sections: List[Dict[str, str]] = []
    current_title = "General"
    current_lines: List[str] = []

    heading_pattern = re.compile(r'^(?:#{1,4}\s+|[A-Z0-9\s\-_]{3,40}:\s*$)(.*)')

    for line in lines:
        match = heading_pattern.match(line)
        if match and len(line.strip()) > 3:
            if current_lines:
                sec_text = "\n".join(current_lines).strip()
                if sec_text:
                    sections.append({"section": current_title, "content": sec_text})
                current_lines = []
            extracted_title = match.group(1).strip() if match.group(1) else line.strip().strip("#:")
            current_title = extracted_title or "General"
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        sec_text = "\n".join(current_lines).strip()
        if sec_text:
            sections.append({"section": current_title, "content": sec_text})

    if not sections:
        sections.append({"section": "General", "content": text.strip()})

    return sections


def split_into_token_sized_chunks(
    text: str,
    max_chars: int = 2000,
    overlap_chars: int = 250,
) -> List[str]:
    """
    Split longer text blocks into overlapping chunks using LangChain RecursiveCharacterTextSplitter.
    """
    if len(text) <= max_chars:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


class StructureAwareChunker:
    """Chunker that preserves section boundaries and enriches metadata from product catalog."""

    def __init__(self, catalog: Optional[Dict[str, Dict[str, Any]]] = None):
        self.catalog = catalog or {}
        self.settings = get_settings()
        # Approx 4 chars per token: 500 tokens ≈ 2000 chars, 60 tokens ≈ 240 chars
        self.max_chars = self.settings.chunk_size_tokens * 4
        self.overlap_chars = self.settings.chunk_overlap_tokens * 4

    def chunk_page(self, page: RawDocumentPage) -> List[Chunk]:
        """Convert a RawDocumentPage into one or more grounded Chunks."""
        chunks: List[Chunk] = []
        raw_text = page.text.strip()
        if not raw_text:
            return []

        # Inferred product metadata
        prod_id = page.inferred_product_id
        prod_name = page.inferred_product_name
        catalog_entry = None

        if prod_id and prod_id in self.catalog:
            catalog_entry = self.catalog[prod_id]
        elif prod_name:
            for pid, entry in self.catalog.items():
                if entry.get("product_name", "").lower() == prod_name.lower():
                    prod_id = pid
                    catalog_entry = entry
                    break

        # If structured format, keep as a single chunk if within limits
        if page.source_type == "structured":
            sections = [{"section": "Catalog Entry", "content": raw_text}]
        else:
            sections = split_text_by_headings(raw_text)

        chunk_counter = 1
        for sec in sections:
            sec_title = sec["section"]
            sec_content = sec["content"]

            sub_chunks = split_into_token_sized_chunks(
                sec_content,
                max_chars=self.max_chars,
                overlap_chars=self.overlap_chars,
            )

            for text_block in sub_chunks:
                # Build context header per Section 6.2
                header_parts = []
                p_display = f"{prod_name} ({prod_id})" if prod_name and prod_id else (prod_name or prod_id or "")
                if p_display:
                    header_parts.append(f"Product: {p_display}")
                header_parts.append(f"Document: {page.document_name}")
                header_parts.append(f"Page: {page.page_number}")
                if sec_title and sec_title != "General":
                    header_parts.append(f"Section: {sec_title}")

                header = " | ".join(header_parts)
                prefixed_text = f"[{header}]\n{text_block}"

                chunk_id = generate_deterministic_chunk_id(
                    prod_id,
                    page.document_name,
                    page.page_number,
                    chunk_counter,
                    prefixed_text,
                )

                # Assemble metadata
                category = page.extra_metadata.get("category") or (catalog_entry.get("category") if catalog_entry else None)
                brand = page.extra_metadata.get("brand") or (catalog_entry.get("brand") if catalog_entry else None)
                supplier = page.extra_metadata.get("supplier") or (catalog_entry.get("supplier") if catalog_entry else None)
                country = page.extra_metadata.get("country") or (catalog_entry.get("country") if catalog_entry else None)
                actual_pname = prod_name or (catalog_entry.get("product_name") if catalog_entry else None)

                metadata = ChunkMetadata(
                    product_id=prod_id,
                    product_name=actual_pname,
                    category=category,
                    brand=brand,
                    supplier=supplier,
                    country=country,
                    document=page.document_name,
                    page=page.page_number,
                    section=sec_title if sec_title != "General" else None,
                    source_type="ocr" if (page.source_type == "ocr" or (sec_title and "Embedded Scanned Image" in sec_title)) else page.source_type,
                    ocr_confidence=page.ocr_confidence,
                    low_confidence=page.low_confidence,
                )

                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=prefixed_text,
                    metadata=metadata,
                )
                chunks.append(chunk)
                chunk_counter += 1

        return chunks

