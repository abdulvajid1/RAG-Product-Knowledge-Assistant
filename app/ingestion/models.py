from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    supplier: Optional[str] = None
    country: Optional[str] = None
    document: str
    page: int = 1
    section: Optional[str] = None
    source_type: str = "text"  # "text", "ocr", "structured"
    ocr_confidence: Optional[float] = None
    low_confidence: bool = False


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.model_dump(),
        }


class RawDocumentPage(BaseModel):
    document_name: str
    page_number: int
    text: str
    source_type: str  # "text", "ocr", "structured"
    ocr_confidence: Optional[float] = None
    low_confidence: bool = False
    inferred_product_id: Optional[str] = None
    inferred_product_name: Optional[str] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

