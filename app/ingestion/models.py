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


from langchain_core.documents import Document


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

    def to_document(self) -> Document:
        """Convert chunk into a standard LangChain Document."""
        return Document(
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                **self.metadata.model_dump(),
            },
        )

    @classmethod
    def from_document(cls, doc: Document) -> "Chunk":
        """Reconstruct Chunk from a LangChain Document."""
        meta = dict(doc.metadata)
        chunk_id = str(meta.pop("chunk_id", ""))
        return cls(
            chunk_id=chunk_id,
            text=doc.page_content,
            metadata=ChunkMetadata(**meta),
        )


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

