from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ProductFilter(BaseModel):
    category: Optional[str] = None
    brand: Optional[str] = None
    supplier: Optional[str] = None
    country: Optional[str] = None
    product_id: Optional[str] = None


class AskRequest(BaseModel):
    query: str = Field(..., description="The user query text")
    filters: Optional[ProductFilter] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query cannot be empty or whitespace only")
        return trimmed


class SourceItem(BaseModel):
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    document: Optional[str] = None
    page: Optional[int] = None
    chunk_id: str
    source_type: Optional[str] = None  # "text", "ocr", "structured"
    score: Optional[float] = None
    text: Optional[str] = None  # optional chunk preview for expandable cards


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)


class ComponentHealth(BaseModel):
    status: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, ComponentHealth]


class FilterOptionsResponse(BaseModel):
    categories: List[str] = Field(default_factory=list)
    brands: List[str] = Field(default_factory=list)
    suppliers: List[str] = Field(default_factory=list)
    countries: List[str] = Field(default_factory=list)
    products: List[Dict[str, str]] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail

