import logging
from typing import List
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider Configuration
    llm_provider: str = Field(default="mock", alias="LLM_PROVIDER")
    llm_model: str = Field(default="mock-model", alias="LLM_MODEL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")

    # Embeddings & Vector Store
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    vector_store_path: str = Field(default="./data/index", alias="VECTOR_STORE_PATH")

    # Retrieval Configuration
    retrieval_mode: str = Field(default="hybrid", alias="RETRIEVAL_MODE")
    top_k: int = Field(default=5, alias="TOP_K")
    score_threshold: float = Field(default=0.35, alias="SCORE_THRESHOLD")
    max_context_chars: int = Field(default=8000, alias="MAX_CONTEXT_CHARS")

    # Chunking Configuration
    chunk_size_tokens: int = Field(default=500, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=60, alias="CHUNK_OVERLAP_TOKENS")

    # OCR Configuration
    ocr_min_text_chars: int = Field(default=30, alias="OCR_MIN_TEXT_CHARS")
    ocr_low_conf_word: float = Field(default=30.0, alias="OCR_LOW_CONF_WORD")
    ocr_low_conf_chunk: float = Field(default=60.0, alias="OCR_LOW_CONF_CHUNK")
    tesseract_cmd: str = Field(default="", alias="TESSERACT_CMD")
    ocr_extract_embedded_images: bool = Field(default=True, alias="OCR_EXTRACT_EMBEDDED_IMAGES")
    ocr_min_image_dim: int = Field(default=100, alias="OCR_MIN_IMAGE_DIM")
    ocr_dpi: int = Field(default=300, alias="OCR_DPI")
    ocr_force_pdf_ocr: bool = Field(default=False, alias="OCR_FORCE_PDF_OCR")

    # API & Security
    max_query_chars: int = Field(default=1000, alias="MAX_QUERY_CHARS")
    cors_origins: str = Field(default="http://localhost:8000,http://127.0.0.1:8000", alias="CORS_ORIGINS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def cors_origins_list(self) -> List[str]:
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return list(self.cors_origins)


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()


def setup_logging() -> None:
    """Configure unified logging for the application."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

